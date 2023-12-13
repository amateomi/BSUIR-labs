use std::{
    collections::HashMap,
    fs,
    io::{self, Read},
    net::{IpAddr, Shutdown, SocketAddr, TcpListener, UdpSocket},
    os::unix::fs::FileExt,
    str,
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

#[derive(Debug, PartialEq, Eq, Hash)]
enum CurrentOperation {
    Download,
    Upload,
    None,
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct ClientInfo {
    download_file_info: Option<FileLoadInfo>,
    upload_file_info: Option<FileLoadInfo>,
    operation: CurrentOperation,
}

pub fn run(protocol: Protocol, ip: Option<IpAddr>) {
    let mut clients = HashMap::<IpAddr, ClientInfo>::new();
    let ip = match ip {
        Some(ip) => ip,
        None => local_ip().expect("Server don't have network interface"),
    };
    let address = (ip, PORT);

    match protocol {
        Protocol::Tcp => {
            let listener = TcpListener::bind(address).expect("Binding should be available");
            info!("Server is listening on address: {address:?}");

            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let peer_ip = stream.peer_addr().unwrap().ip();
                        info!("New peer IP: {}", peer_ip);

                        clients.entry(peer_ip).or_insert(ClientInfo {
                            download_file_info: None,
                            upload_file_info: None,
                            operation: CurrentOperation::None,
                        });
                        handle_connection(&mut Transport::Tcp(stream), &mut clients, peer_ip)
                            .unwrap_or_else(|error| {
                                error!("Connection error: {error}");
                            });
                    }
                    Err(error) => error!("TCP stream error: {error}"),
                }
            }
        }
        Protocol::Udp => {
            let mut transport =
                Transport::Udp(UdpSocket::bind(address).expect("Binding should be available"));
            info!("Server bind on address: {address:?}");

            loop {
                match read_command(&mut transport) {
                    Ok(command) => match command {
                        Command::Echo(payload) => {
                            let message = str::from_utf8(&payload).unwrap().trim_end_matches('\0');
                            let peer_address = message.parse::<SocketAddr>().unwrap();
                            if let Transport::Udp(socket) = &transport {
                                socket
                                    .connect(peer_address)
                                    .expect("Failed to connect to {peer_address}");
                            }
                            let peer_ip = peer_address.ip();

                            info!("New peer IP: {}", peer_ip);
                            clients.entry(peer_ip).or_insert(ClientInfo {
                                download_file_info: None,
                                upload_file_info: None,
                                operation: CurrentOperation::None,
                            });
                            handle_connection(&mut transport, &mut clients, peer_ip)
                                .unwrap_or_else(|error| {
                                    error!("Connection error: {error}");
                                });
                        }
                        _ => {
                            error!("Unexpected command: {command}")
                        }
                    },
                    Err(error) => error!("UDP socket error: {error}"),
                }
            }
        }
    }
}

fn handle_connection(
    transport: &mut Transport,
    clients: &mut HashMap<IpAddr, ClientInfo>,
    peer_ip: IpAddr,
) -> io::Result<()> {
    loop {
        let command = read_command(transport)?;
        match command {
            Command::Echo(payload) => {
                let echo_argument = std::str::from_utf8(&payload).unwrap();
                println!("Echo: {echo_argument}");
            }
            Command::TimeRequest => {
                let time = Local::now().to_rfc2822();
                let response =
                    Command::TimeResponse([0; PAYLOAD_SIZE]).fill_payload(time.as_bytes());
                write_command(transport, response)?;
            }
            Command::Close => {
                match transport {
                    Transport::Tcp(stream) => stream.shutdown(Shutdown::Both)?,
                    Transport::Udp(_) => {}
                }
                return Ok(());
            }
            Command::DownloadRequest(payload) => {
                let file_name = std::str::from_utf8(&payload)
                    .unwrap()
                    .trim_end_matches('\0');
                match &clients.get(&peer_ip).unwrap().download_file_info {
                    Some(file_info) if file_info.name == file_name => {
                        write_command(transport, Command::LoadResumeAvailable)?;
                    }
                    _ => {
                        match fs::File::open(file_name) {
                            Ok(mut file) => {
                                let mut content = Vec::new();
                                file.read_to_end(&mut content).unwrap();
                                clients.entry(peer_ip).and_modify(|peer| {
                                    peer.operation = CurrentOperation::Download;
                                    peer.download_file_info = Some(FileLoadInfo {
                                        name: file_name.to_string(),
                                        last_block_id: 0,
                                        content,
                                    })
                                });
                                write_command(transport, Command::FirstBlockToLoad(0))?;
                            }
                            Err(error) => {
                                error!("Failed to open \"{file_name}\": {error}");
                                write_command(transport, Command::FileNotExist)?;
                            }
                        };
                    }
                }
            }
            Command::UploadRequest(payload) => {
                let file_name = std::str::from_utf8(&payload)
                    .unwrap()
                    .trim_end_matches('\0');
                match &clients.get(&peer_ip).unwrap().upload_file_info {
                    Some(file_info) if file_info.name == file_name => {
                        write_command(transport, Command::LoadResumeAvailable)?
                    }
                    _ => {
                        clients.entry(peer_ip).and_modify(|peer| {
                            peer.operation = CurrentOperation::Upload;
                            peer.upload_file_info = Some(FileLoadInfo {
                                name: file_name.to_string(),
                                last_block_id: 0,
                                content: Vec::new(),
                            })
                        });
                        write_command(transport, Command::FirstBlockToLoad(0))?;
                    }
                }
            }
            Command::LoadResumeAvailable => todo!(),
            Command::LoadResumeResponse(is_resume) => {
                let file_info = clients.get_mut(&peer_ip).unwrap();
                let file_info = match file_info.operation {
                    CurrentOperation::Download => file_info.download_file_info.as_mut().unwrap(),
                    CurrentOperation::Upload => file_info.upload_file_info.as_mut().unwrap(),
                    CurrentOperation::None => panic!("Invalid state"),
                };

                if !is_resume {
                    file_info.last_block_id = 0;
                }
                write_command(
                    transport,
                    Command::FirstBlockToLoad(file_info.last_block_id),
                )?;
            }
            Command::FirstBlockToLoad(_) => todo!(),
            Command::FirstBlockToLoadAcknowledgement => {
                let file_info = clients.get(&peer_ip).unwrap();
                match file_info.operation {
                    CurrentOperation::Download => {
                        let file_info = file_info.download_file_info.as_ref().unwrap();
                        let begin = file_info.last_block_id * PAYLOAD_SIZE;
                        let end = ((file_info.last_block_id + 1) * PAYLOAD_SIZE)
                            .min(file_info.content.len());
                        let block = &file_info.content[begin..end];
                        let response = Command::Load(file_info.last_block_id, [0; PAYLOAD_SIZE])
                            .fill_payload(block);
                        write_command(transport, response)?;
                    }
                    CurrentOperation::Upload => {
                        write_command(transport, Command::FirstBlockToLoadAcknowledgement)?;
                    }
                    CurrentOperation::None => panic!("Invalid state"),
                };
            }
            Command::Load(block_id, payload) => {
                let file_info = clients.get_mut(&peer_ip).unwrap();
                let file_info = file_info.upload_file_info.as_mut().unwrap();

                if block_id != file_info.last_block_id {
                    error!("Unexpected block ID: {block_id}")
                } else {
                    if block_id == 0 {
                        fs::File::create(&file_info.name)?;
                    }
                    let offset = (file_info.last_block_id * PAYLOAD_SIZE) as u64;
                    let file = fs::File::options().write(true).open(&file_info.name)?;
                    file.write_all_at(&payload, offset)?;
                    file.sync_data()?;
                    write_command(
                        transport,
                        Command::LoadAcknowledgement(file_info.last_block_id),
                    )?;
                    file_info.last_block_id += 1;
                }
            }
            Command::LoadAcknowledgement(block_id) => {
                let file_info = clients.get_mut(&peer_ip).unwrap();
                match file_info.operation {
                    CurrentOperation::Download => {
                        let file_info = file_info.download_file_info.as_mut().unwrap();
                        if block_id != file_info.last_block_id {
                            error!("Unexpected block ID: {block_id}");
                        } else {
                            file_info.last_block_id += 1;

                            let begin = file_info.last_block_id * PAYLOAD_SIZE;
                            if begin > file_info.content.len() {
                                write_command(
                                    transport,
                                    Command::LoadFinish(file_info.content.len()),
                                )?;
                                clients.entry(peer_ip).and_modify(|peer| {
                                    peer.operation = CurrentOperation::None;
                                    peer.download_file_info = None
                                });
                            } else {
                                let end = ((file_info.last_block_id + 1) * PAYLOAD_SIZE)
                                    .min(file_info.content.len());
                                let block = &file_info.content[begin..end];
                                let response =
                                    Command::Load(file_info.last_block_id, [0; PAYLOAD_SIZE])
                                        .fill_payload(block);
                                write_command(transport, response)?;
                            }
                        }
                    }
                    CurrentOperation::Upload => {}
                    CurrentOperation::None => panic!("Invalid state"),
                };
            }
            Command::LoadFinish(file_size) => {
                let file_info = clients.get_mut(&peer_ip).unwrap();
                let file_info = file_info.upload_file_info.as_mut().unwrap();

                fs::File::options()
                    .write(true)
                    .open(&file_info.name)?
                    .set_len(file_size as u64)?;

                clients.entry(peer_ip).and_modify(|peer| {
                    peer.operation = CurrentOperation::None;
                    peer.upload_file_info = None
                });
            }
            _ => {
                error!("Unexpected command: {command}");
            }
        }
    }
}
