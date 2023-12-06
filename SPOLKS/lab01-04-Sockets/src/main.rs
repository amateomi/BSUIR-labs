use std::{ process::exit, net::IpAddr };

use log::{ info, error };
use common::*;

mod common;
mod server;
mod client;

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let protocol = cli.protocol;
    info!("Transport layer protocol: {protocol:?}");
    let mode = cli.mode;
    info!("Operation mode: {mode:?}");
    let server_ip = cli.server_ip;
    info!("Server IP: {server_ip:?}");
    if mode == Mode::Client && server_ip.is_none() {
        error!("Must specify server IP in client mode");
        exit(1);
    }
    let server_ip: Option<IpAddr> = match server_ip {
        Some(ip) => Some(ip.parse().expect("Invalid IP format: {ip}")),
        None => None,
    };
    match mode {
        Mode::Server => server::run(protocol, server_ip),
        Mode::Client => client::run(protocol, server_ip.unwrap()),
    };
}
