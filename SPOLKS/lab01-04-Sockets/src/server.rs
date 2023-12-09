use std::{
    collections::HashMap,
    fs,
    io::Read,
    net::{IpAddr, Shutdown, TcpListener, TcpStream},
};

use chrono::Local;
use local_ip_address::local_ip;
use log::{error, info};

use crate::common::*;

#[derive(PartialEq, Eq, Hash, Debug)]
struct FileLoadInfo {
    name: String,
    last_block_id: BlockID,
    content: Vec<u8>,
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct ClientInfo {
    download_file_info: Option<FileLoadInfo>,
    upload_file_info: Option<FileLoadInfo>,
}

pub fn run(_protocol: Protocol, ip: Option<IpAddr>) {
    let mut clients = HashMap::<IpAddr, ClientInfo>::new();
    let ip = match ip {
        Some(ip) => ip,
        None => local_ip().expect("Server don't have network interface"),
    };
    let address = (ip, PORT);
    info!("Server is listening on address: {address:?}");
    let listener = TcpListener::bind(address).expect("Binding should be available");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let peer_ip = stream.peer_addr().unwrap().ip();
                info!("New peer address: {}", peer_ip);
                clients.entry(peer_ip).or_insert(ClientInfo {
                    download_file_info: None,
                    upload_file_info: None,
                });
                handle_stream(stream, &mut clients, peer_ip).unwrap_or_else(|error| {
                    error!("Socket error: {error}");
                });
            }
            Err(error) => error!("TCP stream error: {error}"),
        }
    }
}

fn handle_stream(
    mut stream: TcpStream,
    clients: &mut HashMap<IpAddr, ClientInfo>,
    peer_ip: IpAddr,
) -> Result<(), std::io::Error> {
    loop {
        let command = read_command(&mut stream)?;
        match command {
            Command::Echo(payload) => {
                let echo_argument = std::str::from_utf8(&payload).unwrap();
                println!("Echo: {echo_argument}");
            }
            Command::TimeRequest => {
                let time = Local::now().to_rfc2822();
                let response =
                    Command::TimeResponse([0; PAYLOAD_SIZE]).fill_payload(time.as_bytes());
                write_command(&mut stream, response)?;
            }
            Command::Close => {
                stream.shutdown(Shutdown::Both)?;
                return Ok(());
            }
            Command::DownloadRequest(payload) => {
                let file_name = std::str::from_utf8(&payload)
                    .unwrap()
                    .trim_end_matches('\0');
                match &clients.get(&peer_ip).unwrap().download_file_info {
                    Some(file_info) if file_info.name == file_name => {
                        write_command(&mut stream, Command::LoadResumeAvailable)?;
                    }
                    _ => {
                        match fs::File::open(file_name) {
                            Ok(mut file) => {
                                let mut content = Vec::new();
                                file.read_to_end(&mut content).unwrap();
                                clients.entry(peer_ip).and_modify(|peer| {
                                    peer.download_file_info = Some(FileLoadInfo {
                                        name: file_name.to_string(),
                                        last_block_id: 0,
                                        content,
                                    })
                                });
                                write_command(&mut stream, Command::FirstBlockToLoad(0))?;
                            }
                            Err(error) => {
                                error!("Failed to open \"{file_name}\": {error}");
                                write_command(&mut stream, Command::FileNotExist)?;
                            }
                        };
                    }
                }
            }
            Command::UploadRequest(_) => todo!(),
            Command::LoadResumeAvailable => todo!(),
            Command::LoadResumeResponse(_) => todo!(),
            Command::FirstBlockToLoad(_) => todo!(),
            Command::FirstBlockToLoadAcknowledgement => {
                let file_info = clients.get(&peer_ip).unwrap();
                let file_info = file_info.download_file_info.as_ref().unwrap();

                let begin = file_info.last_block_id * PAYLOAD_SIZE;
                let end =
                    ((file_info.last_block_id + 1) * PAYLOAD_SIZE).min(file_info.content.len());
                let block = &file_info.content[begin..end];
                let response =
                    Command::Load(file_info.last_block_id, [0; PAYLOAD_SIZE]).fill_payload(block);
                write_command(&mut stream, response)?;
            }
            Command::Load(_, _) => todo!(),
            Command::LoadAcknowledgement(block_id) => {
                let file_info = clients.get_mut(&peer_ip).unwrap();
                let file_info = file_info.download_file_info.as_mut().unwrap();

                if block_id != file_info.last_block_id {
                    error!("Unexpected block ID: {block_id}");
                } else {
                    file_info.last_block_id += 1;

                    let begin = file_info.last_block_id * PAYLOAD_SIZE;
                    if begin > file_info.content.len() {
                        write_command(&mut stream, Command::LoadFinish(file_info.content.len()))?;
                    } else {
                        let end = ((file_info.last_block_id + 1) * PAYLOAD_SIZE)
                            .min(file_info.content.len());
                        let block = &file_info.content[begin..end];
                        let response = Command::Load(file_info.last_block_id, [0; PAYLOAD_SIZE])
                            .fill_payload(block);
                        write_command(&mut stream, response)?;
                    }
                }
            }
            Command::LoadFinish(_) => todo!(),
            _ => {
                error!("Unexpected command: {command}");
            }
        }
    }
}
