use log::{error, warn};
use std::{
    fs,
    io::{self, Read},
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

fn main_loop(stream: &mut TcpStream) -> Result<(), io::Error> {
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
                let current_file_name = std::str::from_utf8(&file_name)
                    .unwrap()
                    .trim_end_matches('\0')
                    .to_string();
                write_command(stream, command)?;
                let mut response = read_command(stream)?;
                if response == Command::FileNotExist {
                    error!("File not exist");
                    continue;
                }
                if response == Command::LoadResumeAvailable {
                    write_command(stream, Command::LoadResumeResponse(is_resume_load()))?;
                    response = read_command(stream)?;
                }
                match response {
                    Command::FirstBlockToLoad(_) => {
                        handle_download(
                            response,
                            stream,
                            &current_file_name,
                            &mut current_block_id,
                        )?;
                    }
                    _ => {
                        error!("Unexpected command: {response}")
                    }
                }
            }
            Command::UploadRequest(file_name) => {
                let current_file_name = std::str::from_utf8(&file_name)
                    .unwrap()
                    .trim_end_matches('\0')
                    .to_string();
                match fs::File::open(&current_file_name) {
                    Ok(mut file) => {
                        let mut content = Vec::new();
                        file.read_to_end(&mut content).unwrap();

                        write_command(stream, command)?;

                        let mut response = read_command(stream)?;
                        if response == Command::LoadResumeAvailable {
                            write_command(stream, Command::LoadResumeResponse(is_resume_load()))?;
                            response = read_command(stream)?;
                        }
                        match response {
                            Command::FirstBlockToLoad(block_id) => {
                                current_block_id = block_id;
                                write_command(stream, Command::FirstBlockToLoadAcknowledgement)?;
                                match read_command(stream)? {
                                    Command::FirstBlockToLoadAcknowledgement => {
                                        handle_upload(stream, content, &mut current_block_id)?
                                    }
                                    command => {
                                        error!("Unexpected command: {command}");
                                    }
                                }
                            }
                            _ => {
                                error!("Unexpected command: {response}");
                            }
                        }
                    }
                    Err(error) => {
                        error!("Failed to open \"{current_file_name}\": {error}");
                    }
                };
            }
            _ => {
                error!("Unexpected command: {command}")
            }
        }
    }
}

fn parse_input() -> Command {
    loop {
        for line in io::stdin().lines() {
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

fn is_resume_load() -> bool {
    println!("Load resume available. Enter `1` to resume:");
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(_) => match line.trim().parse::<i32>() {
            Ok(number) => return number == 1,
            Err(error) => {
                warn!("Failed to parse \"{line}\": {error}");
            }
        },
        Err(error) => {
            warn!("Failed to read user input: {error}")
        }
    }
    false
}

fn handle_download(
    command: Command,
    stream: &mut TcpStream,
    current_file_name: &str,
    current_block_id: &mut usize,
) -> Result<(), io::Error> {
    match command {
        Command::FirstBlockToLoad(block_id) => {
            if block_id == 0 {
                fs::File::create(current_file_name)?;
            }
            *current_block_id = block_id;
            write_command(stream, Command::FirstBlockToLoadAcknowledgement)?;

            loop {
                match read_command(stream)? {
                    Command::Load(block_id, payload) => {
                        if *current_block_id != block_id {
                            error!("Unexpected block ID: {block_id}");
                            exit(1);
                        }
                        let offset = (*current_block_id * PAYLOAD_SIZE) as u64;

                        let file = fs::File::options().write(true).open(current_file_name)?;
                        file.write_all_at(&payload, offset)?;
                        file.sync_data()?;

                        write_command(stream, Command::LoadAcknowledgement(*current_block_id))?;

                        *current_block_id += 1;
                    }
                    Command::LoadFinish(file_size) => {
                        fs::File::options()
                            .write(true)
                            .open(current_file_name)?
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
            error!("Unsupported command");
        }
    }
    Ok(())
}

fn handle_upload(
    stream: &mut TcpStream,
    content: Vec<u8>,
    current_block_id: &mut usize,
) -> Result<(), io::Error> {
    loop {
        let begin = *current_block_id * PAYLOAD_SIZE;
        if begin > content.len() {
            write_command(stream, Command::LoadFinish(content.len()))?;
            break;
        } else {
            let end = ((*current_block_id + 1) * PAYLOAD_SIZE).min(content.len());
            let block = &content[begin..end];
            let response = Command::Load(*current_block_id, [0; PAYLOAD_SIZE]).fill_payload(block);
            write_command(stream, response)?;
        }
        loop {
            match read_command(stream)? {
                Command::LoadAcknowledgement(block_id) => {
                    if block_id != *current_block_id {
                        error!("Unexpected block ID: {block_id}")
                    } else {
                        *current_block_id += 1;
                        break;
                    }
                }
                command => {
                    error!("Unexpected command: {command}");
                    exit(1);
                }
            }
        }
    }
    Ok(())
}
