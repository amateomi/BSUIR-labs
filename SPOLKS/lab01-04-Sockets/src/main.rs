use std::{net::IpAddr, process::exit};

use common::*;
use log::{error, info};

mod client;
mod common;
mod server;

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
    let server_ip: Option<IpAddr> =
        server_ip.map(|ip| ip.parse().expect("Invalid IP format: {ip}"));
    match mode {
        Mode::Server => server::run(protocol, server_ip),
        Mode::Client => client::run(protocol, server_ip.unwrap()),
    };
}
