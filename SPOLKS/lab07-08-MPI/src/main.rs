use mpi::{
    environment::Universe,
    traits::{Communicator, Destination, Source},
};
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

const MATRIX_SIZE: usize = 440;

fn main() {
    let mut context = Context::default();
    context.run();
}

struct Context {
    universe: Universe,
    rows_per_slave: usize,
    x: Matrix<f64, MATRIX_SIZE>,
    y: Matrix<f64, MATRIX_SIZE>,
    z: Matrix<f64, MATRIX_SIZE>,
}

impl Default for Context {
    fn default() -> Self {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let slave_count: usize = (world.size() - 1) as usize;

        let remainder = MATRIX_SIZE % slave_count;
        if remainder != 0 {
            eprintln!("slave_count={slave_count} must be devisor of matrix_size={MATRIX_SIZE}, but now division remainder={remainder}");
            world.abort(1);
        }
        let rows_per_slave = MATRIX_SIZE / slave_count;

        let (x, y) = if world.rank() == 0 {
            (
                Matrix::new(Uniform::new(0.0, 1.0)),
                Matrix::new(Uniform::new(0.0, 1.0)),
            )
        } else {
            (Matrix::default(), Matrix::default())
        };

        Self {
            universe,
            rows_per_slave,
            x,
            y,
            z: Matrix::<f64, MATRIX_SIZE>::default(),
        }
    }
}

impl Context {
    fn run(&mut self) {
        if self.universe.world().rank() == 0 {
            let start_time = mpi::time();
            self.run_master();
            let end_time = mpi::time();
            println!("Elapsed time: {:.3}sec", end_time - start_time);
        } else {
            self.run_slave();
        }
    }

    #[cfg(mpi_exec_mode = "sync")]
    fn run_master(&mut self) {
        let world = self.universe.world();

        let chunk_size = self.rows_per_slave * MATRIX_SIZE;

        for (i, chunk) in self.x.data.chunks_exact(chunk_size).enumerate() {
            let rank = (i + 1) as mpi::Rank;
            world.process_at_rank(rank).send(chunk);
            world.process_at_rank(rank).send(&self.y.data);
        }

        // Slaves computation

        for (i, chunk) in self.z.data.chunks_exact_mut(chunk_size).enumerate() {
            let rank = (i + 1) as mpi::Rank;
            world.process_at_rank(rank).receive_into(chunk);
        }
    }

    #[cfg(mpi_exec_mode = "sync")]
    fn run_slave(&mut self) {
        let world = self.universe.world();

        let chunk_index = (world.rank() - 1) as usize;
        let chunk_size = self.rows_per_slave * MATRIX_SIZE;

        let start_index = chunk_index * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        world
            .process_at_rank(0)
            .receive_into(&mut self.x.data[range.clone()]);
        world.process_at_rank(0).receive_into(&mut self.y.data);

        for z_index in range.clone() {
            for i in 0..MATRIX_SIZE {
                let x_index = z_index / MATRIX_SIZE * MATRIX_SIZE + i;
                let y_index = z_index % MATRIX_SIZE + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        world.process_at_rank(0).send(&self.z.data[range]);
    }

    #[cfg(mpi_exec_mode = "async")]
    fn run_master(&mut self) {
        let world = self.universe.world();
        let slave_count = (world.size() - 1) as usize;

        let chunk_size = self.rows_per_slave * MATRIX_SIZE;

        mpi::request::multiple_scope(3 * slave_count, |scope, requests| {
            for (i, chunk) in self.x.data.chunks_exact(chunk_size).enumerate() {
                let rank = (i + 1) as mpi::Rank;
                requests.add(world.process_at_rank(rank).immediate_send(scope, chunk));
                requests.add(
                    world
                        .process_at_rank(rank)
                        .immediate_send(scope, &self.y.data),
                );
            }

            // Slaves computation

            for (i, chunk) in self.z.data.chunks_exact_mut(chunk_size).enumerate() {
                let rank = (i + 1) as mpi::Rank;
                requests.add(
                    world
                        .process_at_rank(rank)
                        .immediate_receive_into(scope, chunk),
                );
            }
            let mut result = vec![];
            requests.wait_all(&mut result);
        });
    }

    #[cfg(mpi_exec_mode = "async")]
    fn run_slave(&mut self) {
        use mpi::request::{multiple_scope, scope, WaitGuard};

        let world = self.universe.world();
        let slave_count = (world.size() - 1) as usize;

        let chunk_index = (world.rank() - 1) as usize;
        let chunk_size = self.rows_per_slave * MATRIX_SIZE;

        let start_index = chunk_index * chunk_size;
        let end_index = start_index + chunk_size;
        let range = start_index..end_index;

        multiple_scope(3 * slave_count, |scope, requests| {
            requests.add(
                world
                    .process_at_rank(0)
                    .immediate_receive_into(scope, &mut self.x.data[range.clone()]),
            );
            requests.add(
                world
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
                world
                    .process_at_rank(0)
                    .immediate_send(scope, &self.z.data[range]),
            );
        })
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
    fn new<D>(distribution: D) -> Self
    where
        D: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        Self {
            data: (0..(SIZE * SIZE))
                .map(|_| rng.sample(&distribution))
                .collect(),
        }
    }
}
