use std::{
    env,
    io::{ErrorKind, Read},
    net::{IpAddr, SocketAddr},
    thread::sleep,
    time::{Duration, Instant},
};

use dns_lookup::lookup_host;
use pnet::packet::{
    icmp::{
        checksum,
        echo_reply::EchoReplyPacket,
        echo_request::{EchoRequestPacket, MutableEchoRequestPacket},
        IcmpCode, IcmpPacket, IcmpTypes,
    },
    ipv4::Ipv4Packet,
    Packet,
};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

const READ_TIMEOUT: Duration = Duration::new(5, 0);
const IDENTIFIER: u16 = 1337;

fn main() -> Result<(), std::io::Error> {
    let target_ip = match env::args().nth(1) {
        Some(arg) => {
            match arg.parse::<IpAddr>() {
                Ok(ip_address) => ip_address,
                Err(ip_parse_error) => {
                    match lookup_host(&arg) {
                        Ok(ip_addresses) => ip_addresses[0],
                        Err(dns_lookup_error) => panic!("Failed to determine target IP address.
                        Argument={arg}, parse error={ip_parse_error}, DNS lookup error={dns_lookup_error}"),
                    }
                },
            }
        },
        None => panic!("First argument must be target to ping/traceroute"),
    };
    if !target_ip.is_ipv4() {
        panic!("Target IP: {target_ip} is not IPv4 address");
    }
    println!("Target IP: {target_ip}");

    let socket_address = SocketAddr::new(target_ip, 0);
    let socket_address = SockAddr::from(socket_address);

    let mut socket = Socket::new(Domain::IPV4, Type::RAW, Some(Protocol::ICMPV4))?;
    socket.set_broadcast(true)?;
    socket.set_read_timeout(Some(READ_TIMEOUT))?;

    let icmp_request_packet_size = EchoRequestPacket::minimum_packet_size();
    let ip_header_size = Ipv4Packet::minimum_packet_size();
    let icmp_reply_packet_size = EchoReplyPacket::minimum_packet_size();

    let mut icmp_send_buffer = vec![0u8; icmp_request_packet_size];
    let mut ip_with_icmp_recv_buffer = vec![0u8; ip_header_size + icmp_reply_packet_size];

    let mut request_sequence_number = 0;
    loop {
        fill_echo_packet(&mut icmp_send_buffer, request_sequence_number);
        let start_time = Instant::now();
        socket.send_to(&icmp_send_buffer, &socket_address)?;
        match socket.read_exact(&mut ip_with_icmp_recv_buffer) {
            Ok(_) => {
                let end_time = Instant::now();
                let delay = end_time - start_time;

                let icmp_packet =
                    EchoReplyPacket::new(&ip_with_icmp_recv_buffer[ip_header_size..]).unwrap();

                let reply_sequence_number = icmp_packet.get_sequence_number();
                if request_sequence_number == reply_sequence_number {
                    println!(
                        "Target IP: {}, sequence number: {}, delay: {}",
                        target_ip,
                        reply_sequence_number,
                        delay.as_millis()
                    );
                } else {
                    eprintln!("Expect sequence number: {request_sequence_number}, but get {reply_sequence_number}");
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                println!("Read timeout ({}ms) expired", READ_TIMEOUT.as_millis())
            }
            Err(error) => break Err(error),
        }
        request_sequence_number += 1;
        sleep(Duration::new(1, 0));
    }
}

fn fill_echo_packet(buffer: &mut [u8], sequence_number: u16) {
    let mut echo_packet = MutableEchoRequestPacket::new(buffer).unwrap();
    echo_packet.set_icmp_type(IcmpTypes::EchoRequest);
    echo_packet.set_icmp_code(IcmpCode::new(0));
    echo_packet.set_sequence_number(sequence_number);
    echo_packet.set_identifier(IDENTIFIER);
    let echo_checksum = checksum(&IcmpPacket::new(echo_packet.packet()).unwrap());
    echo_packet.set_checksum(echo_checksum);
}
