use std::{
    net::{ SocketAddr, TcpListener, IpAddr, TcpStream },
    ffi::c_int,
    io::{ self, BufRead, Write },
};
use socket2::{ Socket, Domain, Type, Protocol };
use local_ip_address::local_ip;
use chrono::Local;

const PORT: u16 = 8080;
const CONNECTIONS_LIMIT: c_int = 8;

fn main() {
    let my_ip = local_ip().expect("Server must have some IP");

    let listener = create_tcp_listener(my_ip, PORT, CONNECTIONS_LIMIT);
    println!("Server address: {my_ip}:{PORT}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                match stream.peer_addr() {
                    Ok(peer_address) => println!("New connection. Address: {peer_address}"),
                    Err(error) => println!("Connection error: {error}"),
                }
                handle_connection(stream);
            }
            Err(error) => {
                eprintln!("Error occurred: {error}");
            }
        }
    }
}

fn create_tcp_listener(ip: IpAddr, port: u16, connections_limit: c_int) -> TcpListener {
    let socket_address = SocketAddr::new(ip, port);
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect(
        "Socket should be creatable"
    );
    socket
        .bind(&socket_address.into())
        .unwrap_or_else(|_| panic!("Socket should be bindable with address: {socket_address}"));
    socket
        .listen(connections_limit)
        .unwrap_or_else(|_|
            panic!("Socket should be available to listen up to  {CONNECTIONS_LIMIT} connections")
        );
    socket.into()
}

fn handle_connection(stream: TcpStream) {
    let mut reader = io::BufReader::new(&stream);
    let mut packet = String::new();

    while reader.read_line(&mut packet).expect("Reading should not fail") != 0 {
        packet.pop();

        let tokens: Vec<_> = packet.split_whitespace().collect();

        if let Some(command_name) = tokens.first() {
            let parameters = &tokens[1..];
            match *command_name {
                "echo" => {
                    if let Err(error) = handle_command_echo(&mut &stream, parameters) {
                        eprintln!("Echo error: {error}");
                    }
                }
                "time" => {
                    if let Err(error) = handle_command_time(&mut &stream) {
                        eprintln!("Time error: {error}");
                    }
                }
                "close" => {
                    if let Err(error) = stream.shutdown(std::net::Shutdown::Both) {
                        eprintln!(
                            "Failed to close connection to {}: {error}",
                            stream.peer_addr().unwrap()
                        );
                    }
                }
                _ => (),
            }
        }

        packet.clear();
    }

    println!("Lost connection to {}", stream.peer_addr().unwrap());
}

fn handle_command_echo(stream: &mut &TcpStream, parameters: &[&str]) -> Result<(), io::Error> {
    stream.write_all(parameters.join(" ").as_bytes())
}

fn handle_command_time(stream: &mut &TcpStream) -> Result<(), io::Error> {
    let time = Local::now().to_rfc2822();
    stream.write_all(time.as_bytes())
}
