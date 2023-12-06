use std::{ net::{ IpAddr, TcpStream }, io::{ Write, self } };

use log::{ error, warn };

use crate::common::*;

pub fn run(_protocol: Protocol, ip: IpAddr) {
    let address = (ip, PORT);
    let mut stream = TcpStream::connect(address).expect("Failed to connect to server");

    loop {
        let command = parse_input();
        let raw_buffer = get_memory(&command);
        match command {
            Command::Echo(_) => {
                stream.write_all(raw_buffer).map_err(|error| error!("Socket write error: {error}"));
            }
            _ => {}
        }
    }
}

fn parse_input() -> Command {
    loop {
        for line in io::stdin().lines() {
            match line {
                Ok(line) if !line.is_empty() => {
                    if let Some((token, parameters)) = line.split_once(' ') {
                        match token {
                            "echo" => {
                                let mut payload: Payload = [0; 1024];
                                payload[0..parameters.len()].clone_from_slice(
                                    parameters.as_bytes()
                                );
                                return Command::Echo(payload);
                            }
                            _ => {
                                warn!("Unknown command: {token}");
                            }
                        }
                    } else {
                        warn!("Line don't contains whitespace");
                    }
                }
                Err(error) => error!("Failed to read input: {error}"),
                _ => {}
            }
        }
    }
}
