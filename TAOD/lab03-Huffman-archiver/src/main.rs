use clap::{Args, Parser};
use std::fs::File;
use std::io::{Read, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    input_file: PathBuf,

    #[command(flatten)]
    action: ActionArg,

    output_file: PathBuf,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct ActionArg {
    /// Encode file
    #[arg(short = 'c')]
    encode: bool,

    /// Decode file
    #[arg(short = 'x')]
    decode: bool,
}

enum ArchiverType {
    Encoder,
    Decoder,
}

struct ByteStream {
    data: Vec<u8>,
    index: usize,
    bit_offset: u32,
}

impl ByteStream {
    fn create(size: usize) -> Self {
        ByteStream {
            data: vec![0; size],
            index: 0,
            bit_offset: 0,
        }
    }

    fn create_empty() -> Self {
        ByteStream {
            data: Vec::new(),
            index: 0,
            bit_offset: 0,
        }
    }

    fn create_filled(mut file: File) -> Self {
        let capacity = file.metadata().unwrap().len() as usize;
        let mut data = Vec::with_capacity(capacity);
        file.read_to_end(&mut data).unwrap();
        Self {
            data,
            index: 0,
            bit_offset: 0,
        }
    }

    fn put_code(&mut self, code: u8, length: u32) {
        let free_bits = u8::BITS - self.bit_offset;
        if length <= free_bits {
            let bits_per_current_byte = free_bits - length;
            self.data[self.index] |= code << bits_per_current_byte;
        } else {
            let bits_per_current_byte = length - free_bits;
            self.data[self.index] |= code >> bits_per_current_byte;
            let bits_per_next_byte = length - free_bits;
            let mask = ((1usize << bits_per_next_byte) - 1) as u8;
            self.data[self.index + 1] |= (code & mask) << (u8::BITS - bits_per_next_byte);
        }
        self.bit_offset += length;
        self.index += (self.bit_offset / u8::BITS) as usize;
        self.bit_offset %= u8::BITS;
    }

    fn get_byte(&self) -> u8 {
        if self.index == (self.data.len() - 1) {
            return self.data[self.index];
        }
        let encoded_bits = u8::BITS - self.bit_offset;
        let mask = ((1usize << encoded_bits) - 1) as u8;
        let current_byte = self.data[self.index];
        let current_byte_part = current_byte & mask;
        let next_byte = self.data[self.index + 1];
        let next_byte_part = next_byte & !mask;
        (current_byte_part << self.bit_offset) | (next_byte_part >> encoded_bits)
    }

    fn advance_stream(&mut self, length: u32) {
        self.bit_offset += length;
        self.index += (self.bit_offset / u8::BITS) as usize;
        self.bit_offset %= u8::BITS;
    }
}

enum Archiver {
    Encoder(Encoder),
    Decoder(Decoder),
}

fn write_to_file(
    mut output_file: File,
    words_count: u64,
    lookup_code_table: &[Code],
    lookup_length_table: &[u32],
    payload: &[u8],
) {
    output_file.write_all(&words_count.to_le_bytes()).unwrap();

    output_file.write_all(lookup_code_table).unwrap();

    for length in lookup_length_table {
        output_file.write_all(&length.to_le_bytes()).unwrap();
    }

    output_file.write_all(payload).unwrap();
}

fn read_from_file(
    mut input_file: File,
    words_count: &mut u64,
    lookup_code_table: &mut [Code],
    lookup_length_table: &mut [u32],
    payload: &mut Vec<u8>,
) {
    let mut bytes = [0; size_of::<u64>()];
    input_file.read_exact(&mut bytes).unwrap();
    *words_count = u64::from_le_bytes(bytes);

    input_file.read_exact(lookup_code_table).unwrap();
    for length in &mut *lookup_length_table {
        let mut bytes = [0; size_of::<u32>()];
        input_file.read_exact(&mut bytes).unwrap();
        *length = u32::from_le_bytes(bytes);
    }

    input_file.read_to_end(payload).unwrap();
}

struct ArchiverContext {
    input_stream: ByteStream,
    output_stream: ByteStream,
    output_file: File,
    archiver: Archiver,
}

impl ArchiverContext {
    fn new(input_file_path: &Path, output_file_path: &Path, archiver_type: ArchiverType) -> Self {
        let input_file = File::open(input_file_path).unwrap();
        let output_file = File::create(output_file_path).unwrap();
        match archiver_type {
            ArchiverType::Encoder => {
                let input_size = input_file.metadata().unwrap().len() as usize;
                Self {
                    input_stream: ByteStream::create_filled(input_file),
                    output_stream: ByteStream::create(input_size),
                    output_file,
                    archiver: Archiver::Encoder(Encoder::default()),
                }
            }
            ArchiverType::Decoder => {
                let mut decoder = Decoder::default();
                let mut input_stream = ByteStream::create_empty();
                read_from_file(
                    input_file,
                    &mut decoder.words_count,
                    &mut decoder.lookup_code_table,
                    &mut decoder.lookup_length_table,
                    &mut input_stream.data,
                );
                Self {
                    input_stream,
                    output_stream: ByteStream::create(decoder.words_count as usize),
                    output_file,
                    archiver: Archiver::Decoder(decoder),
                }
            }
        }
    }

    fn run(mut self) {
        match self.archiver {
            Archiver::Encoder(ref mut encoder) => {
                encoder.count_frequency(&self.input_stream.data);
                encoder.create_tree();
                encoder.compute_node_depths();
                encoder.compute_code_depths();
                encoder.reevaluate_code_depths();
                encoder.create_codes();
                encoder.prepare_for_lookup();
                for byte in &self.input_stream.data {
                    let code = encoder.lookup_code_table[*byte as usize];
                    let length = encoder.lookup_length_table[*byte as usize];
                    self.output_stream.put_code(code, length);
                }
                let payload_size = self.output_stream.index + 1;
                write_to_file(
                    self.output_file,
                    self.input_stream.data.len() as u64,
                    &encoder.lookup_code_table,
                    &encoder.lookup_length_table,
                    &self.output_stream.data[0..payload_size],
                );
            }
            Archiver::Decoder(ref mut decoder) => {
                decoder.create_decode_tables();
                for i in 0..decoder.words_count {
                    let key = self.input_stream.get_byte();
                    let word = decoder.decode_word_table[key as usize];
                    let length = decoder.decode_length_table[key as usize];
                    self.output_stream.data[i as usize] = word;
                    self.input_stream.advance_stream(length);
                }
                self.output_file
                    .write_all(&self.output_stream.data)
                    .unwrap();
            }
        }
    }
}

const BYTE_COUNT: usize = (u8::MAX as usize) + 1;
const MAX_CODE_LENGTH: usize = 8;

type Word = u8;
type Code = u8;

struct Encoder {
    words: Vec<Word>,
    table: Vec<u32>,
    codes: Vec<Code>,
    lookup_code_table: Vec<Code>,
    lookup_length_table: Vec<u32>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            words: Vec::with_capacity(BYTE_COUNT),
            table: Vec::with_capacity(BYTE_COUNT),
            codes: Vec::with_capacity(BYTE_COUNT),
            lookup_code_table: vec![0; BYTE_COUNT],
            lookup_length_table: vec![0; BYTE_COUNT],
        }
    }
}

impl Encoder {
    fn count_frequency(&mut self, data: &[u8]) {
        let mut freq_table = [0u32; BYTE_COUNT];
        for byte in data {
            freq_table[*byte as usize] += 1;
        }
        for (word, freq) in freq_table.iter().enumerate() {
            if *freq != 0 {
                self.words.push(word as u8);
                self.table.push(*freq);
                self.codes.push(0);
            }
        }
        let mut permutation = permutation::sort(&self.table);
        permutation.apply_slice_in_place(&mut self.words);
        permutation.apply_slice_in_place(&mut self.table);
    }

    fn create_tree(&mut self) {
        let n = self.table.len();
        let mut s = 0;
        let mut r = 0;
        for t in 0..(n - 1) {
            if (s > (n - 1)) || (r < t && self.table[r] < self.table[s]) {
                self.table[t] = self.table[r];
                self.table[r] = (t + 1) as u32;
                r += 1;
            } else {
                self.table[t] = self.table[s];
                s += 1;
            }
            if (s > (n - 1)) || (r < t && self.table[r] < self.table[s]) {
                self.table[t] += self.table[r];
                self.table[r] = (t + 1) as u32;
                r += 1;
            } else {
                self.table[t] += self.table[s];
                s += 1;
            }
        }
    }

    fn compute_node_depths(&mut self) {
        let n = self.table.len();
        if n < 3 {
            for depth in self.table.iter_mut() {
                *depth = 1;
            }
            return;
        }
        self.table[n - 2] = 0;
        for t in (0..(n - 2)).rev() {
            self.table[t] = self.table[(self.table[t] - 1) as usize] + 1;
        }
    }

    fn compute_code_depths(&mut self) {
        let n = self.table.len();
        if n < 3 {
            return;
        }
        let mut a = 1;
        let mut u = 0;
        let mut d = 0;
        let mut t: i64 = (n - 2) as i64;
        let mut x: i64 = (n - 1) as i64;
        while a > 0 {
            while t >= 0 && self.table[t as usize] == d {
                u += 1;
                t -= 1;
            }
            while a > u {
                self.table[x as usize] = d;
                x -= 1;
                a -= 1;
            }
            a = u * 2;
            d += 1;
            u = 0;
        }
    }

    fn reevaluate_code_depths(&mut self) {
        let n = self.table.len();
        let l = self.table[0] as usize;
        let mut m = vec![0; l.max(MAX_CODE_LENGTH) + 1];
        for i in 0..n {
            m[self.table[i] as usize] += 1;
        }
        for i in ((1 + MAX_CODE_LENGTH)..(l + 1)).rev() {
            while m[i] > 0 {
                let mut j = i - 1;
                while {
                    j -= 1;
                    m[j] <= 0
                } {}
                m[i] -= 2;
                m[i - 1] += 1;
                m[j + 1] += 2;
                m[j] -= 1;
            }
        }
        let mut n = 0;
        for i in (1..(MAX_CODE_LENGTH + 1)).rev() {
            let mut k = m[i];
            while k > 0 {
                self.table[n] = i as u32;
                n += 1;
                k -= 1;
            }
        }
    }

    fn create_codes(&mut self) {
        let n = self.table.len();
        let l = self.table[0] as usize;
        let mut m = vec![0; l + 1];
        for i in 0..n {
            m[self.table[i] as usize] += 1;
        }
        let mut s: u32 = 0;
        let mut base = vec![0; l + 1];
        for k in (1..l + 1).rev() {
            base[k] = s >> (l - k);
            s += m[k] << (l - k);
        }
        let mut p = 0;
        let mut j = 0;
        for i in 0..(n) {
            if p != self.table[i] {
                j = 0;
                p = self.table[i];
            }
            self.codes[i] = (j + base[self.table[i] as usize]) as u8;
            j += 1;
        }
    }

    fn prepare_for_lookup(&mut self) {
        let n = self.words.len();
        for i in 0..n {
            let word = self.words[i];
            let length = self.table[i];
            let code = self.codes[i];
            self.lookup_code_table[word as usize] = code;
            self.lookup_length_table[word as usize] = length;
        }
    }
}

struct Decoder {
    words_count: u64,
    lookup_code_table: Vec<Code>,
    lookup_length_table: Vec<u32>,
    decode_word_table: Vec<Word>,
    decode_length_table: Vec<u32>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            words_count: 0,
            lookup_code_table: vec![0; BYTE_COUNT],
            lookup_length_table: vec![0; BYTE_COUNT],
            decode_word_table: vec![0; BYTE_COUNT],
            decode_length_table: vec![0; BYTE_COUNT],
        }
    }
}

impl Decoder {
    fn create_decode_tables(&mut self) {
        for word in 0..BYTE_COUNT {
            let code = self.lookup_code_table[word];
            let length = self.lookup_length_table[word];
            if length == 0 {
                continue;
            }
            let left_bits = u8::BITS - length;
            let min_key_to_match = code << left_bits;
            let max_key_to_match = min_key_to_match | ((1 << left_bits) - 1);
            for key in min_key_to_match..=max_key_to_match {
                self.decode_word_table[key as usize] = word as u8;
                self.decode_length_table[key as usize] = length;
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let archiver_type = if cli.action.encode {
        ArchiverType::Encoder
    } else {
        ArchiverType::Decoder
    };
    ArchiverContext::new(&cli.input_file, &cli.output_file, archiver_type).run();
}
