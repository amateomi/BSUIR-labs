pub use bytes_cast::BytesCast;
pub use clap::{ Parser, ValueEnum };

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

pub type Payload = [u8; 1024];
pub type FileName = [u8; 256];
pub type BlockID = usize;

pub enum Command {
    Echo(Payload),
    Time,
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
    LoadFinish,
}

pub fn get_memory<T>(input: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(input as *const _ as *const u8, std::mem::size_of::<T>()) }
}

pub fn get_memory_mut<T>(input: &mut T) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(input as *mut _ as *mut u8, std::mem::size_of::<T>()) }
}

pub fn clone_into_array<A, T>(slice: &[T]) -> A where A: Sized + Default + AsMut<[T]>, T: Clone {
    let mut a = Default::default();
    <A as AsMut<[T]>>::as_mut(&mut a).clone_from_slice(slice);
    a
}
