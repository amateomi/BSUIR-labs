use log::error;
use std::{
    fs,
    net::{IpAddr, TcpStream},
    os::unix::fs::FileExt,
    process::exit,
    str::{self, FromStr},
};

use crate::common::*;

pub fn run(_protocol: Protocol, ip: IpAddr) {
    let address = (ip, PORT);
    let mut stream = TcpStream::connect(address).expect("Failed to connect to server");

    loop {
        if let Err(error) = main_loop(&mut stream) {
            error!("Error: {error}");
        }
    }
}

fn main_loop(stream: &mut TcpStream) -> Result<(), std::io::Error> {
    let mut current_file_name = String::new();
    let mut current_block_id = 0;
    loop {
        let command = parse_input();
        match command {
            Command::Echo(_) => {
                write_command(stream, command)?;
            }
            Command::TimeRequest => {
                write_command(stream, command)?;
                let response = read_command(stream)?;
                match response {
                    Command::TimeResponse(payload) => {
                        let time = str::from_utf8(&payload).unwrap();
                        println!("Time: {time}");
                    }
                    _ => error!("Unexpected command: {response}"),
                }
            }
            Command::Close => {
                write_command(stream, command)?;
                exit(0);
            }
            Command::DownloadRequest(file_name) => {
                current_file_name = std::str::from_utf8(&file_name)
                    .unwrap()
                    .trim_end_matches('\0')
                    .to_string();
                write_command(stream, command)?;
                let response = read_command(stream)?;
                match response {
                    Command::FileNotExist => {
                        error!("File not exist")
                    }
                    Command::LoadResumeAvailable => todo!(),
                    Command::FirstBlockToLoad(block_id) => {
                        if block_id == 0 {
                            fs::File::create(&current_file_name)?;
                        }
                        current_block_id = block_id;
                        write_command(stream, Command::FirstBlockToLoadAcknowledgement)?;

                        loop {
                            match read_command(stream)? {
                                Command::Load(block_id, payload) => {
                                    if current_block_id != block_id {
                                        error!("Unexpected block ID: {block_id}");
                                        exit(1);
                                    }
                                    let offset = (current_block_id * PAYLOAD_SIZE) as u64;

                                    let file =
                                        fs::File::options().write(true).open(&current_file_name)?;
                                    file.write_all_at(&payload, offset)?;
                                    file.sync_data()?;

                                    write_command(
                                        stream,
                                        Command::LoadAcknowledgement(current_block_id),
                                    )?;

                                    current_block_id += 1;
                                }
                                Command::LoadFinish(file_size) => {
                                    fs::File::options()
                                        .write(true)
                                        .open(&current_file_name)?
                                        .set_len(file_size as u64)?;
                                    break;
                                }
                                command => {
                                    error!("Unexpected command: {command}");
                                    exit(1);
                                }
                            };
                        }
                    }
                    _ => {
                        error!("Unexpected command: {response}")
                    }
                }
            }
            _ => {
                error!("Unexpected command: {command}")
            }
        }
    }
}

fn parse_input() -> Command {
    loop {
        for line in std::io::stdin().lines() {
            match line {
                Ok(line) => {
                    if line.is_empty() {
                        continue;
                    }
                    match Command::from_str(&line) {
                        Ok(command) => return command,
                        Err(error) => error!("Parse error: {error}"),
                    }
                }
                Err(error) => error!("Failed to read input: {error}"),
            }
        }
    }
}
