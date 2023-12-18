use local_ip_address::linux::local_ip;
use log::{error, warn};
use std::{
    fs,
    io::{self, Read},
    net::{IpAddr, SocketAddr, TcpStream, UdpSocket},
    os::unix::fs::FileExt,
    process::exit,
    str::{self, FromStr},
};

use crate::common::*;

pub fn run(protocol: Protocol, ip: IpAddr) {
    let address = (ip, PORT);
    let mut transport = match protocol {
        Protocol::Tcp => {
            Transport::Tcp(TcpStream::connect(address).expect("Failed to connect to server"))
        }
        Protocol::Udp => {
            let my_ip = local_ip().unwrap();
            let my_address = SocketAddr::new(my_ip, 0);
            let socket = UdpSocket::bind(my_address).expect("Failed to bind to {my_address}");
            socket
                .connect(address)
                .expect("Failed to connect to server");

            let start_command = format!("echo {}", socket.local_addr().unwrap())
                .parse::<Command>()
                .expect("Failed to parse start command: {start_command}");

            let mut transport = Transport::Udp(socket);
            write_command(&mut transport, start_command).expect("Failed to send my ip to server");
            transport
        }
    };
    loop {
        if let Err(error) = main_loop(&mut transport) {
            error!("Error: {error}");
        }
    }
}

fn main_loop(transport: &mut Transport) -> io::Result<()> {
    let mut current_block_id = 0;
    loop {
        let command = parse_input();
        match command {
            Command::Echo(_) => {
                write_command(transport, command)?;
            }
            Command::TimeRequest => {
                write_command(transport, command)?;
                let response = read_command(transport)?;
                match response {
                    Command::TimeResponse(payload) => {
                        let time = str::from_utf8(&payload).unwrap();
                        println!("Time: {time}");
                    }
                    _ => error!("Unexpected command: {response}"),
                }
            }
            Command::Close => {
                write_command(transport, command)?;
                exit(0);
            }
            Command::DownloadRequest(file_name) => {
                let current_file_name = std::str::from_utf8(&file_name)
                    .unwrap()
                    .trim_end_matches('\0')
                    .to_string();
                write_command(transport, command)?;
                let mut response = read_command(transport)?;
                if response == Command::FileNotExist {
                    error!("File not exist");
                    continue;
                }
                if response == Command::LoadResumeAvailable {
                    write_command(transport, Command::LoadResumeResponse(is_resume_load()))?;
                    response = read_command(transport)?;
                }
                match response {
                    Command::FirstBlockToLoad(_) => {
                        handle_download(
                            response,
                            transport,
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

                        write_command(transport, command)?;

                        let mut response = read_command(transport)?;
                        if response == Command::LoadResumeAvailable {
                            write_command(
                                transport,
                                Command::LoadResumeResponse(is_resume_load()),
                            )?;
                            response = read_command(transport)?;
                        }
                        match response {
                            Command::FirstBlockToLoad(block_id) => {
                                current_block_id = block_id;
                                write_command(transport, Command::FirstBlockToLoadAcknowledgement)?;
                                match read_command(transport)? {
                                    Command::FirstBlockToLoadAcknowledgement => {
                                        handle_upload(transport, content, &mut current_block_id)?
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
    transport: &mut Transport,
    current_file_name: &str,
    current_block_id: &mut usize,
) -> io::Result<()> {
    match command {
        Command::FirstBlockToLoad(block_id) => {
            if block_id == 0 {
                fs::File::create(current_file_name)?;
            }
            *current_block_id = block_id;
            write_command(transport, Command::FirstBlockToLoadAcknowledgement)?;

            loop {
                match read_command(transport)? {
                    Command::Load(block_id, payload) => {
                        if *current_block_id != block_id {
                            error!("Unexpected block ID: {block_id}");
                            exit(1);
                        }
                        let offset = (*current_block_id * PAYLOAD_SIZE) as u64;

                        let file = fs::File::options().write(true).open(current_file_name)?;
                        file.write_all_at(&payload, offset)?;
                        file.sync_data()?;

                        write_command(transport, Command::LoadAcknowledgement(*current_block_id))?;

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
    transport: &mut Transport,
    content: Vec<u8>,
    current_block_id: &mut usize,
) -> io::Result<()> {
    loop {
        let begin = *current_block_id * PAYLOAD_SIZE;
        if begin > content.len() {
            write_command(transport, Command::LoadFinish(content.len()))?;
            break;
        } else {
            let end = ((*current_block_id + 1) * PAYLOAD_SIZE).min(content.len());
            let block = &content[begin..end];
            let response = Command::Load(*current_block_id, [0; PAYLOAD_SIZE]).fill_payload(block);
            write_command(transport, response)?;
        }
        loop {
            match read_command(transport)? {
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
