use std::{
    env,
    io::{ErrorKind, Read, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    thread::sleep,
    time::{Duration, Instant},
};

use dns_lookup::lookup_host;
use local_ip_address::local_ip;
use pnet::packet::{
    icmp::{
        checksum,
        echo_reply::EchoReplyPacket,
        echo_request::{EchoRequestPacket, IcmpCodes, MutableEchoRequestPacket},
        time_exceeded::TimeExceededPacket,
        IcmpPacket, IcmpType, IcmpTypes,
    },
    ip::IpNextHeaderProtocols,
    ipv4::{Ipv4Packet, MutableIpv4Packet},
    Packet, PrimitiveValues,
};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};

const READ_TIMEOUT: Duration = Duration::new(5, 0);
const IDENTIFIER: u16 = 1337;

enum Mode {
    Ping,
    Traceroute,
    Smurf,
}

fn main() {
    let target_ip = get_target_ip();
    println!("Target IP: {target_ip}");
    let (socket, socket_address) = get_icmp_socket(target_ip);
    match get_mode() {
        Mode::Ping => ping(socket, socket_address),
        Mode::Traceroute => traceroute(socket, socket_address),
        Mode::Smurf => smurf(socket, socket_address),
    }
}

fn get_target_ip() -> IpAddr {
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
    target_ip
}

fn get_icmp_socket(target_ip: IpAddr) -> (Socket, SockAddr) {
    let socket_address = SocketAddr::new(target_ip, 0);
    let socket_address = SockAddr::from(socket_address);

    let socket = Socket::new(Domain::IPV4, Type::RAW, Some(Protocol::ICMPV4)).unwrap();
    socket.set_broadcast(true).unwrap();
    socket.set_read_timeout(Some(READ_TIMEOUT)).unwrap();

    (socket, socket_address)
}

fn get_mode() -> Mode {
    match env::args().nth(2) {
        Some(arg) => match arg.as_str() {
            "ping" => Mode::Ping,
            "trace" => Mode::Traceroute,
            "smurf" => Mode::Smurf,
            _ => {
                panic!("Unknown mode: {arg}")
            }
        },
        None => Mode::Ping,
    }
}

fn ping(mut socket: Socket, socket_address: SockAddr) {
    socket.connect(&socket_address).unwrap();

    let icmp_request_packet_size = EchoRequestPacket::minimum_packet_size();
    let ip_header_size = Ipv4Packet::minimum_packet_size();
    let icmp_reply_packet_size = EchoReplyPacket::minimum_packet_size();

    let mut icmp_send_buffer = vec![0u8; icmp_request_packet_size];
    let mut ip_with_icmp_recv_buffer = vec![0u8; ip_header_size + icmp_reply_packet_size];

    let mut request_sequence_number = 0;
    loop {
        fill_echo_packet(&mut icmp_send_buffer, request_sequence_number);
        let start_time = Instant::now();
        socket.write_all(&icmp_send_buffer).unwrap();
        match socket.read_exact(&mut ip_with_icmp_recv_buffer) {
            Ok(_) => {
                let end_time = Instant::now();
                let delay = end_time - start_time;

                let icmp_packet =
                    EchoReplyPacket::new(&ip_with_icmp_recv_buffer[ip_header_size..]).unwrap();

                let reply_sequence_number = icmp_packet.get_sequence_number();
                if request_sequence_number == reply_sequence_number {
                    println!(
                        "Sequence number: {}, delay: {}",
                        reply_sequence_number,
                        delay.as_millis()
                    );
                } else {
                    println!("Expect sequence number: {request_sequence_number}, but get {reply_sequence_number}");
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                println!("Read timeout ({}ms) expired", READ_TIMEOUT.as_millis())
            }
            Err(error) => panic!("{error}"),
        }
        request_sequence_number += 1;
        sleep(Duration::new(1, 0));
    }
}

fn traceroute(mut socket: Socket, socket_address: SockAddr) {
    let icmp_request_packet_size = EchoRequestPacket::minimum_packet_size();
    let ip_header_size = Ipv4Packet::minimum_packet_size();
    let icmp_reply_packet_size = EchoReplyPacket::minimum_packet_size();
    let icmp_time_exceeded_packet_size = TimeExceededPacket::minimum_packet_size();

    let mut icmp_send_buffer = vec![0u8; icmp_request_packet_size];
    let mut ip_with_icmp_recv_buffer =
        vec![0u8; ip_header_size + icmp_reply_packet_size.max(icmp_time_exceeded_packet_size)];

    let mut request_sequence_number = 0;
    let mut time_to_live = 1;
    loop {
        socket.set_ttl(time_to_live).unwrap();
        fill_echo_packet(&mut icmp_send_buffer, request_sequence_number);
        socket.send_to(&icmp_send_buffer, &socket_address).unwrap();
        match socket.read_exact(&mut ip_with_icmp_recv_buffer) {
            Ok(_) => {
                let icmp_type = IcmpType::new(ip_with_icmp_recv_buffer[ip_header_size]);
                match icmp_type {
                    IcmpTypes::TimeExceeded | IcmpTypes::EchoReply => {
                        let ip_packet = Ipv4Packet::new(&ip_with_icmp_recv_buffer).unwrap();
                        println!(
                            "Time to live: {}, source: {}",
                            time_to_live,
                            ip_packet.get_source()
                        );
                    }
                    _ => panic!(
                        "Unexpected ICMP type: {}",
                        icmp_type.to_primitive_values().0
                    ),
                }
                if icmp_type == IcmpTypes::EchoReply {
                    return;
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                println!("Read timeout ({}ms) expired", READ_TIMEOUT.as_millis())
            }
            Err(error) => panic!("Failed to read ICMP packet: {error}"),
        }
        request_sequence_number += 1;
        time_to_live += 1;
    }
}

fn smurf(socket: Socket, socket_address: SockAddr) {
    socket.set_header_included(true).unwrap();

    let icmp_request_packet_size = EchoRequestPacket::minimum_packet_size();
    let ip_header_size = Ipv4Packet::minimum_packet_size();

    let mut ip_with_icmp_send_buffer = vec![0u8; ip_header_size + icmp_request_packet_size];

    let mut sequence_number = 0;
    loop {
        fill_ip_header(
            &mut ip_with_icmp_send_buffer,
            *socket_address.as_socket_ipv4().unwrap().ip(),
        );
        fill_echo_packet(
            &mut ip_with_icmp_send_buffer[ip_header_size..],
            sequence_number,
        );
        socket
            .send_to(&ip_with_icmp_send_buffer, &socket_address)
            .unwrap();
        sequence_number = sequence_number.wrapping_add(1);
    }
}

fn fill_ip_header(buffer: &mut [u8], target_ip: Ipv4Addr) {
    let total_packet_length = buffer.len() as u16;
    let local_ip = match local_ip().unwrap() {
        IpAddr::V4(local_ip) => local_ip,
        IpAddr::V6(_) => panic!("Impossible situation"),
    };
    let mut ip_packet = MutableIpv4Packet::new(buffer).unwrap();
    ip_packet.set_version(4);
    ip_packet.set_header_length((Ipv4Packet::minimum_packet_size() / 4) as u8);
    ip_packet.set_dscp(0);
    ip_packet.set_ecn(0);
    ip_packet.set_total_length(total_packet_length);
    ip_packet.set_identification(IDENTIFIER);
    ip_packet.set_flags(0);
    ip_packet.set_fragment_offset(0);
    ip_packet.set_ttl(255);
    ip_packet.set_next_level_protocol(IpNextHeaderProtocols::Icmp);
    ip_packet.set_checksum(0); // Hope Linux will fill it
    ip_packet.set_source(target_ip);
    ip_packet.set_destination(local_ip);
}

fn fill_echo_packet(buffer: &mut [u8], sequence_number: u16) {
    let mut echo_packet = MutableEchoRequestPacket::new(buffer).unwrap();
    echo_packet.set_icmp_type(IcmpTypes::EchoRequest);
    echo_packet.set_icmp_code(IcmpCodes::NoCode);
    echo_packet.set_sequence_number(sequence_number);
    echo_packet.set_identifier(IDENTIFIER);
    let echo_checksum = checksum(&IcmpPacket::new(echo_packet.packet()).unwrap());
    echo_packet.set_checksum(echo_checksum);
}
