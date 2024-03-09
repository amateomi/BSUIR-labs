use std::env;

use mpi::{
    environment::Universe,
    topology::{SimpleCommunicator, UserGroup},
    traits::{Communicator, Destination, Group, Source},
    Rank,
};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    Rng, SeedableRng,
};

const MATRIX_SIZE: usize = 480 * 4;

fn main() {
    let mut context = Context::default();
    context.run();
}

struct GroupInfo {
    id: UserGroup,
    communicator: Option<SimpleCommunicator>,
    slave_count: usize,
    rows_per_slave: usize,
}

struct Context {
    groups: Vec<GroupInfo>,
    _universe: Universe,
    x: Matrix<f64, MATRIX_SIZE>,
    y: Matrix<f64, MATRIX_SIZE>,
    z: Matrix<f64, MATRIX_SIZE>,
}

impl Default for Context {
    fn default() -> Self {
        let group_count = match env::var("mpi_exec_group_count") {
            Ok(group_count) => group_count.parse().unwrap_or(1),
            Err(_) => 1,
        } as usize;
        println!("Group count={group_count}");

        let _universe = mpi::initialize().unwrap();
        let world = _universe.world();

        let mut random: StdRng = SeedableRng::seed_from_u64(3);

        let mut grouped_workers_count = 0;
        let ungrouped_workers: Vec<Rank> = (0..world.size()).collect();

        let workers_per_group_count: Vec<usize> = {
            let random_numbers: Vec<f64> = (0..group_count)
                .map(|_| random.gen_range(0.0..1.0))
                .collect();
            let sum: f64 = random_numbers.iter().sum();
            random_numbers
                .iter()
                .map(|x| {
                    2 + (x / sum * ((ungrouped_workers.len() - 2 * group_count) as f64)) as usize
                })
                .collect()
        };

        let groups: Vec<GroupInfo> = (0..group_count).map(|i| {
            let workers_count = if i < group_count - 1 {
                workers_per_group_count[i]
            } else {
                ungrouped_workers.len() - grouped_workers_count
            };

            let start = grouped_workers_count as Rank;
            let end = ungrouped_workers.len().min(grouped_workers_count + workers_count) as Rank;
            grouped_workers_count += workers_count;
            let ranks_to_include: Vec<Rank> = (start..end).collect();

            let id = world.group().include(&ranks_to_include);
            let communicator = world.split_by_subgroup_collective(&id);

            let mut slave_count = ranks_to_include.len() - 1;
            if cfg!(mpi_exec_mode = "collective") || cfg!(mpi_exec_mode = "file") {
                // In this modes masters are also part of computation
                slave_count += 1;
            };
            let remainder = MATRIX_SIZE % slave_count;
            if remainder != 0 {
                eprintln!("slave_count={slave_count} must be devisor of matrix_size={MATRIX_SIZE}, but now division remainder={remainder}");
                world.abort(1);
            }
            let rows_per_slave = MATRIX_SIZE / slave_count;

            GroupInfo {
                id,
                communicator,
                slave_count,
                rows_per_slave
            }
        }).collect();

        let mut x = Matrix::default();
        let mut y = Matrix::default();
        for (i, group) in groups.iter().enumerate() {
            if let Some(group_id) = group.id.rank() {
                if group_id == 0 {
                    println!("Group {i}: master=1, slaves={}", group.slave_count);
                    x = Matrix::new(&mut random, Uniform::new(0.0, 1.0));
                    y = Matrix::new(&mut random, Uniform::new(0.0, 1.0));
                }
            }
        }

        Self {
            groups,
            _universe,
            x,
            y,
            z: Matrix::<f64, MATRIX_SIZE>::default(),
        }
    }
}

impl Context {
    fn run(&mut self) {
        for (index, group) in self.groups.iter().enumerate() {
            if let Some(group_id) = group.id.rank() {
                if group_id == 0 {
                    let start_time = mpi::time();
                    self.run_master(index);
                    let end_time = mpi::time();
                    println!(
                        "Group {index}: elapsed time is {:.3}sec",
                        end_time - start_time
                    );
                    return;
                }
                self.run_slave(index);
                return;
            }
        }
    }

    #[cfg(mpi_exec_mode = "sync")]
    fn run_master(&mut self, group_index: usize) {
        let communicator = self.groups[group_index].communicator.as_ref().unwrap();

        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        for (i, chunk) in self.x.data.chunks_exact(chunk_size).enumerate() {
            let rank = (i + 1) as mpi::Rank;
            communicator.process_at_rank(rank).send(chunk);
            communicator.process_at_rank(rank).send(&self.y.data);
        }

        // Slaves computation

        for (i, chunk) in self.z.data.chunks_exact_mut(chunk_size).enumerate() {
            let rank = (i + 1) as mpi::Rank;
            communicator.process_at_rank(rank).receive_into(chunk);
        }
    }

    #[cfg(mpi_exec_mode = "sync")]
    fn run_slave(&mut self, group_index: usize) {
        let communicator = self.groups[group_index].communicator.as_ref().unwrap();

        let chunk_index = (communicator.rank() - 1) as usize;
        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        let start_index = chunk_index * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        communicator
            .process_at_rank(0)
            .receive_into(&mut self.x.data[range.clone()]);
        communicator
            .process_at_rank(0)
            .receive_into(&mut self.y.data);

        for z_index in range.clone() {
            for i in 0..MATRIX_SIZE {
                let x_index = z_index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = z_index % MATRIX_SIZE + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        communicator.process_at_rank(0).send(&self.z.data[range]);
    }

    #[cfg(mpi_exec_mode = "async")]
    fn run_master(&mut self, group_index: usize) {
        let communicator = self.groups[group_index].communicator.as_ref().unwrap();
        let slave_count = self.groups[group_index].slave_count;

        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        mpi::request::multiple_scope(3 * slave_count, |scope, requests| {
            for (i, chunk) in self.x.data.chunks_exact(chunk_size).enumerate() {
                let rank = (i + 1) as mpi::Rank;
                requests.add(
                    communicator
                        .process_at_rank(rank)
                        .immediate_send(scope, chunk),
                );
                requests.add(
                    communicator
                        .process_at_rank(rank)
                        .immediate_send(scope, &self.y.data),
                );
            }

            // Slaves computation

            for (i, chunk) in self.z.data.chunks_exact_mut(chunk_size).enumerate() {
                let rank = (i + 1) as mpi::Rank;
                requests.add(
                    communicator
                        .process_at_rank(rank)
                        .immediate_receive_into(scope, chunk),
                );
            }
            let mut result = vec![];
            requests.wait_all(&mut result);
        });
    }

    #[cfg(mpi_exec_mode = "async")]
    fn run_slave(&mut self, group_index: usize) {
        use mpi::request::{multiple_scope, scope, WaitGuard};

        let communicator = self.groups[group_index].communicator.as_ref().unwrap();
        let slave_count = self.groups[group_index].slave_count;

        let chunk_index = (communicator.rank() - 1) as usize;
        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        let start_index = chunk_index * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        multiple_scope(3 * slave_count, |scope, requests| {
            requests.add(
                communicator
                    .process_at_rank(0)
                    .immediate_receive_into(scope, &mut self.x.data[range.clone()]),
            );
            requests.add(
                communicator
                    .process_at_rank(0)
                    .immediate_receive_into(scope, &mut self.y.data),
            );
            let mut result = vec![];
            requests.wait_all(&mut result);
        });

        for z_index in range.clone() {
            for i in 0..MATRIX_SIZE {
                let x_index = z_index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = z_index % MATRIX_SIZE + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        scope(|scope| {
            let _waiter = WaitGuard::from(
                communicator
                    .process_at_rank(0)
                    .immediate_send(scope, &self.z.data[range]),
            );
        })
    }

    #[cfg(mpi_exec_mode = "collective")]
    fn run_master(&mut self, group_index: usize) {
        use mpi::traits::Root;

        let communicator = self.groups[group_index].communicator.as_ref().unwrap();
        let master_process = communicator.process_at_rank(0);

        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        master_process.broadcast_into(&mut self.y.data);

        let mut chunk = vec![0.0; chunk_size];
        master_process.scatter_into_root(&self.x.data, &mut chunk);
        self.x
            .data
            .chunks_exact_mut(chunk_size)
            .nth(0)
            .unwrap()
            .copy_from_slice(&chunk);

        for (index, z) in chunk.iter_mut().enumerate().take(chunk_size) {
            for i in 0..MATRIX_SIZE {
                let x_index = index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = index % MATRIX_SIZE + i * MATRIX_SIZE;
                *z += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        master_process.gather_into_root(&chunk, &mut self.z.data);
    }

    #[cfg(mpi_exec_mode = "collective")]
    fn run_slave(&mut self, group_index: usize) {
        use mpi::traits::Root;

        let communicator = self.groups[group_index].communicator.as_ref().unwrap();
        let master_process = communicator.process_at_rank(0);

        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;

        master_process.broadcast_into(&mut self.y.data);

        let rank = communicator.rank() as usize;
        let chunk = self.x.data.chunks_exact_mut(chunk_size).nth(rank).unwrap();
        master_process.scatter_into(chunk);

        let start_index = rank * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        for z_index in range {
            for i in 0..MATRIX_SIZE {
                let x_index = z_index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = z_index % MATRIX_SIZE + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        let chunk = self.z.data.chunks_exact_mut(chunk_size).nth(rank).unwrap();
        master_process.gather_into(chunk);
    }

    #[cfg(mpi_exec_mode = "file")]
    fn run_master(&mut self, group_index: usize) {
        self.run_file(group_index);
    }

    #[cfg(mpi_exec_mode = "file")]
    fn run_slave(&mut self, group_index: usize) {
        self.run_file(group_index);
    }

    #[cfg(mpi_exec_mode = "file")]
    fn run_file(&mut self, group_index: usize) {
        use std::{
            alloc::{alloc, Layout},
            ffi::{c_int, CString},
            mem, ptr,
        };

        use mpi::{
            ffi::{
                MPI_File, MPI_File_close, MPI_File_open, MPI_File_read_at_all, MPI_File_write,
                MPI_File_write_at_all, MPI_Offset, MPI_Status, MPI_MODE_CREATE, MPI_MODE_RDWR,
                MPI_MODE_WRONLY, MPI_SUCCESS, RSMPI_DOUBLE, RSMPI_INFO_NULL,
            },
            raw::AsRaw,
            traits::Collection,
        };

        let communicator = self.groups[group_index].communicator.as_ref().unwrap();

        let chunk_size = self.groups[group_index].rows_per_slave * MATRIX_SIZE;
        let rank = communicator.rank() as usize;
        let offset = (rank * chunk_size * mem::size_of::<f64>()) as MPI_Offset;

        unsafe {
            let x_file = CString::new(format!("x_{group_index}")).unwrap();
            let mut x_file_handle: MPI_File = alloc(Layout::new::<MPI_File>()).cast();
            if MPI_File_open(
                communicator.as_raw(),
                x_file.as_ptr(),
                (MPI_MODE_CREATE | MPI_MODE_RDWR) as c_int,
                RSMPI_INFO_NULL,
                ptr::addr_of_mut!(x_file_handle),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to open file x");
            }
            let mut status: MPI_Status = mem::zeroed();
            if rank == 0
                && MPI_File_write(
                    x_file_handle,
                    self.x.data.as_ptr().cast(),
                    self.x.data.count(),
                    RSMPI_DOUBLE,
                    ptr::addr_of_mut!(status),
                ) as u32
                    != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to write file x");
            }
            let chunk = self
                .x
                .data
                .chunks_exact_mut(chunk_size)
                .nth(communicator.rank() as usize)
                .unwrap();
            if MPI_File_read_at_all(
                x_file_handle,
                offset,
                chunk.as_mut_ptr().cast(),
                chunk_size as c_int,
                RSMPI_DOUBLE,
                ptr::addr_of_mut!(status),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to read file x");
            }
            if MPI_File_close(ptr::addr_of_mut!(x_file_handle)) as u32 != MPI_SUCCESS {
                panic!("Group {group_index} failed to close file x");
            }
        }
        unsafe {
            let y_file = CString::new(format!("y_{group_index}")).unwrap();
            let mut y_file_handle: MPI_File = alloc(Layout::new::<MPI_File>()).cast();
            if MPI_File_open(
                communicator.as_raw(),
                y_file.as_ptr(),
                (MPI_MODE_CREATE | MPI_MODE_RDWR) as c_int,
                RSMPI_INFO_NULL,
                ptr::addr_of_mut!(y_file_handle),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to open file y");
            }
            let mut status: MPI_Status = mem::zeroed();
            if rank == 0
                && MPI_File_write(
                    y_file_handle,
                    self.y.data.as_ptr().cast(),
                    self.y.data.count(),
                    RSMPI_DOUBLE,
                    ptr::addr_of_mut!(status),
                ) as u32
                    != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to write file y");
            }
            if MPI_File_read_at_all(
                y_file_handle,
                0,
                self.y.data.as_mut_ptr().cast(),
                self.y.data.count(),
                RSMPI_DOUBLE,
                ptr::addr_of_mut!(status),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to read file y");
            }
            if MPI_File_close(ptr::addr_of_mut!(y_file_handle)) as u32 != MPI_SUCCESS {
                panic!("Group {group_index} failed to close file y");
            }
        }

        let start_index = rank * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        for z_index in range {
            for i in 0..MATRIX_SIZE {
                let x_index = z_index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = z_index % MATRIX_SIZE + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        unsafe {
            let z_file = CString::new(format!("z_{group_index}")).unwrap();
            let mut z_file_handle: MPI_File = alloc(Layout::new::<MPI_File>()).cast();
            if MPI_File_open(
                communicator.as_raw(),
                z_file.as_ptr(),
                (MPI_MODE_CREATE | MPI_MODE_WRONLY) as c_int,
                RSMPI_INFO_NULL,
                ptr::addr_of_mut!(z_file_handle),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to open file z");
            }
            let chunk = self
                .z
                .data
                .chunks_exact(chunk_size)
                .nth(communicator.rank() as usize)
                .unwrap();
            let mut status: MPI_Status = mem::zeroed();
            if MPI_File_write_at_all(
                z_file_handle,
                offset,
                chunk.as_ptr().cast(),
                chunk_size as c_int,
                RSMPI_DOUBLE,
                ptr::addr_of_mut!(status),
            ) as u32
                != MPI_SUCCESS
            {
                panic!("Group {group_index} failed to write file z");
            }
            if MPI_File_close(ptr::addr_of_mut!(z_file_handle)) as u32 != MPI_SUCCESS {
                panic!("Group {group_index} failed to close file z");
            }
        }
    }
}

#[derive(Debug)]
struct Matrix<T, const SIZE: usize> {
    data: Vec<T>,
}

impl<T, const SIZE: usize> Default for Matrix<T, SIZE>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self {
            data: vec![T::default(); SIZE * SIZE],
        }
    }
}

impl<T, const SIZE: usize> Matrix<T, SIZE> {
    fn new<D>(random: &mut StdRng, distribution: D) -> Self
    where
        D: Distribution<T>,
    {
        Self {
            data: (0..(SIZE * SIZE))
                .map(|_| random.sample(&distribution))
                .collect(),
        }
    }
}
