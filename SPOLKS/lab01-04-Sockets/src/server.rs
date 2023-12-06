use std::{ net::{ IpAddr, TcpListener, TcpStream }, io::{ Read, self }, mem::MaybeUninit };

use local_ip_address::local_ip;
use log::{ info, error };

use crate::common::*;

pub fn run(_protocol: Protocol, ip: Option<IpAddr>) {
    let ip = match ip {
        Some(ip) => ip,
        None => local_ip().expect("Server don't have network interface"),
    };
    let address = (ip, PORT);
    info!("Server is listening on address: {address:?}");
    let listener = TcpListener::bind(address).expect("Binding should be avaliable");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                info!("Peer address: {}", stream.peer_addr().unwrap());
                handle_stream(stream).unwrap_or_else(|error| {
                    error!("Socket error: {error}");
                });
            }
            Err(error) => error!("TCP stream error: {error}"),
        }
    }
}

fn handle_stream(mut stream: TcpStream) -> Result<(), io::Error> {
    loop {
        let mut command: MaybeUninit<Command> = MaybeUninit::uninit();
        let raw_buffer = get_memory_mut(&mut command);
        stream.read_exact(raw_buffer)?;
        let command = unsafe { command.assume_init() };
        match command {
            Command::Echo(payload) => {
                let echo_argument = std::str::from_utf8(payload.as_bytes()).unwrap();
                println!("Echo: {echo_argument}");
            }
            _ => {}
            // Command::Time
            // Command::TimeResponse(Payload),
            // Command::Close,
            // Command::DownloadRequest(FileName),
            // Command::UploadRequest(FileName),
            // Command::FileNotExist,
            // Command::LoadResumeAvailable,
            // Command::LoadResumeResponse(bool),
            // Command::FirstBlockToLoad(BlockID),
            // Command::FirstBlockToLoadAcknowledgement,
            // Command::Load(BlockID, Payload),
            // Command::LoadAcknowledgement(BlockID),
            // Command::LoadFinish,
        }
    }
}
