use std::{
    net::{ SocketAddr, TcpListener, IpAddr, TcpStream },
    ffi::c_int,
    io::{ self, BufRead, Write, Read },
    env,
    fs::File,
    str::FromStr,
};
use socket2::{ Socket, Domain, Type, Protocol, SockAddr };
use local_ip_address::local_ip;
use chrono::Local;

const PORT: u16 = 8080;
const CONNECTIONS_LIMIT: c_int = 8;

const FILE_TRANSACTION_SIZE: usize = 1024;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        let arg: &str = &args[1];
        match arg {
            "server" => run_server(),
            "client" => {
                if args.len() == 3 {
                    let server_ip: SocketAddr = args[2].parse().unwrap();
                    run_client(server_ip.into());
                } else {
                    eprintln!("Must specify server <ip:port>");
                }
            }
            _ => eprintln!("Must specify one argument: server or client"),
        }
    } else {
        eprintln!("Must specify at least one argument: <server> or <client>");
    }
}

enum Command {
    Echo(Vec<String>),
    Time,
    Close,
    Upload,
    Download(FileTransferInfo),
}

impl FromStr for Command {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "echo" => Ok(Command::Echo(Vec::<String>::new())),
            "time" => Ok(Command::Time),
            "close" => Ok(Command::Close),
            "upload" => Ok(Command::Upload),
            "download" =>
                Ok(
                    Command::Download(FileTransferInfo {
                        client_ip: IpAddr::from_str("0.0.0.0").unwrap(),
                        file_path: String::new(),
                        chunk_id: 0,
                    })
                ),
            _ => Err("Unknown command".to_string()),
        }
    }
}



fn run_client(server_ip: SockAddr) {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect(
        "Socket should be creatable"
    );
    socket.connect(&server_ip).expect("Connection to server should work");
    println!("Connection to server established");

    let mut stream: TcpStream = socket.into();

    for line in io::stdin().lines() {
        match line {
            Ok(line) if !line.is_empty() => {
                let line = line + "\n";
                stream.write_all(line.as_bytes()).unwrap();

                let mut reader = io::BufReader::new(&stream);
                let mut buffer = Vec::<u8>::new();

                match reader.read_until(b'\0', &mut buffer) {
                    Ok(_) => {
                        match String::from_utf8(buffer.clone()) {
                            Ok(text) => println!("{}", text),
                            Err(error) =>
                                eprintln!("Failed to parse buffer to utf8 string: {error}"),
                        }
                        buffer.clear();
                    }
                    Err(error) => eprintln!("Failed to read from server: {error}"),
                }
            }
            Err(error) => eprintln!("Failed to read line: {error}"),
            _ => {}
        }
    }
}

fn run_server() {
    let my_ip = local_ip().expect("Server must have some IP");

    let listener = create_tcp_listener(my_ip, PORT, CONNECTIONS_LIMIT);
    println!("Server address: {my_ip}:{PORT}");

    let mut file_transfer_info: Option<FileTransferInfo> = None;

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                match stream.peer_addr() {
                    Ok(peer_address) => println!("New connection. Address: {peer_address}"),
                    Err(error) => println!("Connection error: {error}"),
                }
                handle_connection(stream, &mut file_transfer_info);
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
    socket.set_reuse_address(true).expect("Port reuse should be available");
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

fn handle_connection(stream: TcpStream, file_transfer_info: &mut Option<FileTransferInfo>) {
    let client_ip = stream.peer_addr().unwrap().ip();

    let mut reader = io::BufReader::new(&stream);
    let mut packet = String::new();

    loop {
        match reader.read_line(&mut packet) {
            Ok(read_size) => {
                if read_size == 0 {
                    break;
                }
                packet.pop();

                let tokens: Vec<_> = packet.split_whitespace().collect();

                if let Some(command_name) = tokens.first() {
                    let command: Command = command_name.parse();
                    let parameters = &tokens[1..];
                    match *command_name {
                        "echo" => {
                            println!("-- echo command");
                            if let Err(error) = handle_command_echo(&mut &stream, parameters) {
                                eprintln!("Echo error: {error}");
                            }
                        }
                        "time" => {
                            println!("-- time command");
                            if let Err(error) = handle_command_time(&mut &stream) {
                                eprintln!("Time error: {error}");
                            }
                        }
                        "close" => {
                            println!("-- close command");
                            if let Err(error) = stream.shutdown(std::net::Shutdown::Both) {
                                eprintln!(
                                    "Failed to close connection to {}: {error}",
                                    stream.peer_addr().unwrap()
                                );
                            }
                        }
                        "upload" => {
                            println!("-- upload command");
                            handle_command_upload();
                        }
                        "download" => {
                            println!("-- download command");
                            if let Some(file_path) = parameters.first() {
                                let mut chunk_id = 0;
                                if let Some(info) = file_transfer_info {
                                    if
                                        info.client_ip == client_ip &&
                                        &info.file_path.as_str() == file_path
                                    {
                                        chunk_id = info.chunk_id;
                                        println!("Restoring download at chunk {chunk_id}");
                                    }
                                }
                                *file_transfer_info = Some(FileTransferInfo {
                                    client_ip,
                                    file_path: file_path.to_string(),
                                    chunk_id,
                                });
                                if
                                    let Err(error) = handle_command_download(
                                        &mut &stream,
                                        file_transfer_info.as_mut().unwrap()
                                    )
                                {
                                    eprintln!("Download error: {error}");
                                } else {
                                    *file_transfer_info = None;
                                }
                            } else {
                                eprintln!(
                                    "Download command parameters length is invalid: {}, should be 1",
                                    parameters.len()
                                );
                            }
                        }
                        _ => (),
                    }
                }
                packet.clear();
            }
            Err(error) => {
                eprintln!("Connection error: {error}");
                break;
            }
        }
    }
    println!("Lost connection to {}", client_ip);
    let _ = stream.shutdown(std::net::Shutdown::Both);
    if let Some(info) = file_transfer_info {
        println!("{:?}", info);
    }
}

fn handle_command_echo(stream: &mut &TcpStream, parameters: &[&str]) -> io::Result<()> {
    let arguments = parameters.join(" ") + "\0";
    stream.write_all(arguments.as_bytes())
}

fn handle_command_time(stream: &mut &TcpStream) -> io::Result<()> {
    let time = Local::now().to_rfc2822() + "\0";
    stream.write_all(time.as_bytes())
}

#[derive(Debug)]
struct FileTransferInfo {
    client_ip: IpAddr,
    file_path: String,
    chunk_id: usize,
}

fn handle_command_upload() {
    todo!()
}

fn handle_command_download(
    stream: &mut &TcpStream,
    file_transfer_info: &mut FileTransferInfo
) -> io::Result<()> {
    let mut file_content = Vec::<u8>::new();

    let size = File::open(file_transfer_info.file_path.as_str())?.read_to_end(&mut file_content)?;
    let chunk_count =
        (((size as f64) / (FILE_TRANSACTION_SIZE as f64)).ceil() as usize) -
        file_transfer_info.chunk_id;

    let header_packet = format!("{chunk_count}\0");
    stream.write_all(header_packet.as_bytes())?;

    let mut i = file_transfer_info.chunk_id * FILE_TRANSACTION_SIZE;
    while i < size {
        println!("chunk: {}", file_transfer_info.chunk_id);

        let end = (i + FILE_TRANSACTION_SIZE).min(size);

        let mut packet: [u8; FILE_TRANSACTION_SIZE + 1] = [0; FILE_TRANSACTION_SIZE + 1];
        packet[..FILE_TRANSACTION_SIZE].clone_from_slice(&file_content[i..end]);
        packet[FILE_TRANSACTION_SIZE] = b'\0';

        stream.write_all(packet.as_slice())?;

        file_transfer_info.chunk_id += 1;
        i = file_transfer_info.chunk_id * FILE_TRANSACTION_SIZE;
    }

    Ok(())
}
