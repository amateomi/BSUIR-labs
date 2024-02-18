use mpi::{
    environment::Universe,
    traits::{Communicator, Destination, Source},
};
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

const MATRIX_SIZE: usize = 1000;

fn main() {
    let mut context = Context::default();
    context.run();
}

struct Context {
    universe: Universe,
    rows_per_slave_count: usize,
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
        let rows_per_slave_count = MATRIX_SIZE / slave_count;

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
            rows_per_slave_count,
            x,
            y,
            z: Matrix::<f64, MATRIX_SIZE>::default(),
        }
    }
}

impl Context {
    fn run(&mut self) {
        if self.universe.world().rank() == 0 {
            self.run_master();
        } else {
            self.run_slave();
        }
    }

    fn run_master(&mut self) {
        let world = self.universe.world();
        for rank in 1..world.size() {
            let start_index = (rank - 1) as usize * self.rows_per_slave_count * MATRIX_SIZE;
            let end_index = start_index + self.rows_per_slave_count * MATRIX_SIZE;
            let range = start_index..end_index;

            world.process_at_rank(rank).send(&self.x.data[range]);
            world.process_at_rank(rank).send(&self.y.data);
        }
        for rank in 1..world.size() {
            let start_index = (rank - 1) as usize * self.rows_per_slave_count * MATRIX_SIZE;
            let end_index = start_index + self.rows_per_slave_count * MATRIX_SIZE;
            let range = start_index..end_index;

            world
                .process_at_rank(rank)
                .receive_into(&mut self.z.data[range]);
        }
        dbg!(&self.x);
        dbg!(&self.y);
        dbg!(&self.z);
    }

    fn run_slave(&mut self) {
        let world = self.universe.world();

        let start_index = (world.rank() - 1) as usize * self.rows_per_slave_count * MATRIX_SIZE;
        let end_index = start_index + self.rows_per_slave_count * MATRIX_SIZE;
        let range = start_index..end_index;

        world
            .process_at_rank(0)
            .receive_into(&mut self.x.data[range.clone()]);

        world.process_at_rank(0).receive_into(&mut self.y.data);

        for z_index in range.clone() {
            for i in 0..MATRIX_SIZE {
                let x_index = (z_index / MATRIX_SIZE) * MATRIX_SIZE + i;
                let y_index = (z_index % MATRIX_SIZE) + i * MATRIX_SIZE;
                self.z.data[z_index] += self.x.data[x_index] * self.y.data[y_index];
            }
        }

        world.process_at_rank(0).send(&self.z.data[range]);
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
