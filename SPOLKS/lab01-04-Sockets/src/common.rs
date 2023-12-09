use std::{
    fmt,
    io::{Read, Write},
    mem::MaybeUninit,
    net::TcpStream,
    str,
};

pub use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Transfer layer protocol to use
    #[arg(short, long, value_enum)]
    pub protocol: Protocol,

    /// Mode for program
    #[arg(short, long, value_enum)]
    pub mode: Mode,

    /// Server IP to connect. Client mode only
    #[arg(short, long)]
    pub server_ip: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Server mode
    Server,
    /// Client mode
    Client,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum Protocol {
    /// Use TCP protocol
    Tcp,
    /// Use UDP protocol
    Udp,
}

pub const PORT: u16 = 6996;
pub const PAYLOAD_SIZE: usize = 1024;
pub const FILE_NAME_SIZE: usize = 256;
pub const DEFAULT_FILE: &str = "bible.txt";

pub type Payload = [u8; PAYLOAD_SIZE];
pub type FileName = [u8; FILE_NAME_SIZE];
pub type BlockID = usize;
pub type FileSize = usize;

#[derive(Debug)]
pub enum Command {
    Echo(Payload),
    TimeRequest,
    TimeResponse(Payload),
    Close,
    DownloadRequest(FileName),
    UploadRequest(FileName),
    FileNotExist,
    LoadResumeAvailable,
    LoadResumeResponse(bool),
    FirstBlockToLoad(BlockID),
    FirstBlockToLoadAcknowledgement,
    Load(BlockID, Payload),
    LoadAcknowledgement(BlockID),
    LoadFinish(FileSize),
}

use log::info;
use Command::*;

impl Command {
    pub fn fill_payload(self, data: &[u8]) -> Self {
        match self {
            TimeRequest
            | Close
            | FileNotExist
            | LoadResumeAvailable
            | FirstBlockToLoadAcknowledgement
            | FirstBlockToLoad(_)
            | LoadAcknowledgement(_)
            | LoadFinish(_) => self,
            Echo(_) | TimeResponse(_) => {
                let mut payload: Payload = [0; PAYLOAD_SIZE];
                payload[0..data.len()].clone_from_slice(data);
                match self {
                    Echo(_) => Echo(payload),
                    TimeResponse(_) => TimeResponse(payload),
                    _ => panic!("Impossible state"),
                }
            }
            DownloadRequest(_) | UploadRequest(_) => {
                let mut file_name: FileName = [0; FILE_NAME_SIZE];
                file_name[0..data.len()].clone_from_slice(data);
                match self {
                    DownloadRequest(_) => DownloadRequest(file_name),
                    UploadRequest(_) => UploadRequest(file_name),
                    _ => panic!("Impossible state"),
                }
            }
            LoadResumeResponse(_) => LoadResumeResponse(*data.first().unwrap() == b'1'),
            Load(block_id, _) => {
                let mut payload: Payload = [0; PAYLOAD_SIZE];
                payload[0..data.len()].clone_from_slice(data);
                Load(block_id, payload)
            }
        }
    }
}

pub struct CommandParseError(String);

impl fmt::Display for CommandParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown command: {}", self.0)
    }
}

impl str::FromStr for Command {
    type Err = CommandParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let line = s.trim();
        let (token, parameters) = line.split_once(' ').unwrap_or((line, ""));
        match token {
            "echo" => Ok(Echo([0; PAYLOAD_SIZE]).fill_payload(parameters.as_bytes())),
            "time" => Ok(Command::TimeRequest),
            "close" => Ok(Command::Close),
            "download" => {
                Ok(DownloadRequest([0; FILE_NAME_SIZE]).fill_payload(parameters.as_bytes()))
            }
            _ => Err(CommandParseError(token.to_string())),
        }
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Echo(payload) => {
                let argument = str::from_utf8(payload).unwrap_or("none UTF8 sequence");
                write!(f, "[echo] \"{argument}\"")
            }
            TimeRequest => {
                write!(f, "[time request]")
            }
            TimeResponse(payload) => {
                let argument = str::from_utf8(payload).unwrap_or("none UTF8 sequence");
                write!(f, "[time response] time=\"{argument}\"")
            }
            Close => {
                write!(f, "[close]")
            }
            DownloadRequest(payload) => {
                let file_name = str::from_utf8(payload).unwrap_or(DEFAULT_FILE);
                write!(f, "[download] file=\"{file_name}\"")
            }
            UploadRequest(payload) => {
                let file_name = str::from_utf8(payload).unwrap_or(DEFAULT_FILE);
                write!(f, "[upload] file=\"{file_name}\"")
            }
            FileNotExist => {
                write!(f, "[file not exist]")
            }
            LoadResumeAvailable => {
                write!(f, "[load resume available]")
            }
            LoadResumeResponse(is_resume_response) => {
                write!(f, "[load resume response] resume={is_resume_response}")
            }
            FirstBlockToLoad(block_id) => {
                write!(f, "[first block to load] blockID={block_id}")
            }
            FirstBlockToLoadAcknowledgement => {
                write!(f, "[first block to load acknowledgement]")
            }
            Load(block_id, payload) => {
                write!(f, "[load] blockID={block_id} payload={payload:?}")
            }
            LoadAcknowledgement(block_id) => {
                write!(f, "[load acknowledgement] blockID={block_id}")
            }
            LoadFinish(file_size) => {
                write!(f, "[load finish] file size={file_size}")
            }
        }
    }
}

pub fn get_memory<T>(input: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(input as *const _ as *const u8, std::mem::size_of::<T>()) }
}

pub fn get_memory_mut<T>(input: &mut T) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(input as *mut _ as *mut u8, std::mem::size_of::<T>()) }
}

pub fn read_command(stream: &mut TcpStream) -> Result<Command, std::io::Error> {
    let mut command: MaybeUninit<Command> = MaybeUninit::uninit();
    let raw_buffer = get_memory_mut(&mut command);
    stream.read_exact(raw_buffer)?;
    let command = unsafe { command.assume_init() };
    info!("Read command: {command}");
    Ok(command)
}

pub fn write_command(stream: &mut TcpStream, command: Command) -> Result<(), std::io::Error> {
    info!("Write command: {command}");
    let raw_buffer = get_memory(&command);
    stream.write_all(raw_buffer)
}
