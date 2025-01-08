use std::fmt::{Display, Result};

#[derive(Clone, Copy)]
enum Cell {
    Blocked,
    Unvisited { id: u32 },
    Visited { id: u32, visit_step: u32 },
}

impl Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        match self {
            Cell::Blocked => write!(f, "[....:....]"),
            Cell::Unvisited { id } => write!(f, "[{:04}:????]", id),
            Cell::Visited { id, visit_step } => write!(f, "[{:04}:{:04}]", id, visit_step),
        }
    }
}

impl Cell {
    fn new(id: u32) -> Self {
        Cell::Unvisited { id }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum MoveDirection {
    Up,
    Right,
    Down,
    Left,
}

impl Display for MoveDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        match self {
            MoveDirection::Up => write!(f, "Up"),
            MoveDirection::Right => write!(f, "Right"),
            MoveDirection::Down => write!(f, "Down"),
            MoveDirection::Left => write!(f, "Left"),
        }
    }
}

type MoveLogic = [MoveDirection; 4];
  
struct Grid<const N: usize, const M: usize, const N_EXTENDED: usize, const M_EXTENDED: usize> {
    grid: [[Cell; N_EXTENDED]; M_EXTENDED],
    current_pos_x: usize,
    current_pos_y: usize,
    move_logic: MoveLogic,
    visit_step: u32,
}

impl<const N: usize, const M: usize, const N_EXTENDED: usize, const M_EXTENDED: usize> Display
    for Grid<N, M, N_EXTENDED, M_EXTENDED>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        for row in self.grid {
            writeln!(f)?;
            for item in row {
                write!(f, "{} ", item)?;
            }
        }
        writeln!(f)?;
        writeln!(
            f,
            "current position: ({},{})",
            self.current_pos_y, self.current_pos_x
        )?;
        writeln!(
            f,
            "move logic: {}->{}->{}->{}",
            self.move_logic[0], self.move_logic[1], self.move_logic[2], self.move_logic[3]
        )?;
        writeln!(f, "visit step: {}", self.visit_step)
    }
}

impl<const N: usize, const M: usize, const N_EXTENDED: usize, const M_EXTENDED: usize>
    Grid<N, M, N_EXTENDED, M_EXTENDED>
{
    fn new(input_grid: [[Cell; N]; M], start_pos: (usize, usize), move_logic: MoveLogic) -> Self {
        let mut grid = [[Cell::Blocked; N_EXTENDED]; M_EXTENDED];
        for y in 0..M {
            for x in 0..N {
                grid[y + 1][x + 1] = input_grid[y][x];
            }
        }

        let (start_pos_y, start_pos_x) = start_pos;
        if start_pos_x >= N {
            panic!("Start x={} position is out of bound", start_pos_x);
        }
        if start_pos_y >= M {
            panic!("Start y={} position is out of bound", start_pos_y);
        }
        let current_pos_x = start_pos_x + 1;
        let current_pos_y = start_pos_y + 1;
        let start_cell = &mut grid[current_pos_y][current_pos_x];
        if let Cell::Unvisited { id } = *start_cell {
            *start_cell = Cell::Visited { id, visit_step: 0 };
        } else {
            panic!("Start cell={} is not unvisited", start_cell);
        }

        let check_direction = |move_direction, move_logic: &MoveLogic| {
            let count = move_logic.iter().filter(|&x| *x == move_direction).count();
            if count != 1 {
                panic!(
                    "{} move direction is present {} times",
                    move_direction, count
                );
            }
        };
        check_direction(MoveDirection::Up, &move_logic);
        check_direction(MoveDirection::Right, &move_logic);
        check_direction(MoveDirection::Down, &move_logic);
        check_direction(MoveDirection::Left, &move_logic);

        Grid {
            grid,
            current_pos_x,
            current_pos_y,
            move_logic,
            visit_step: 0,
        }
    }

    fn make_n_unique_steps(&mut self, step_count: u32) {
        for move_direction in self.move_logic {
            if self.visit_step == step_count {
                return;
            }
            let (next_pos_y, next_pos_x) = match move_direction {
                MoveDirection::Up => (self.current_pos_y - 1, self.current_pos_x),
                MoveDirection::Right => (self.current_pos_y, self.current_pos_x + 1),
                MoveDirection::Down => (self.current_pos_y + 1, self.current_pos_x),
                MoveDirection::Left => (self.current_pos_y, self.current_pos_x - 1),
            };
            let next_cell = &mut self.grid[next_pos_y][next_pos_x];
            if let Cell::Unvisited { id } = *next_cell {
                let prev_pos_x = self.current_pos_x;
                let prev_pos_y = self.current_pos_y;
                self.current_pos_x = next_pos_x;
                self.current_pos_y = next_pos_y;
                self.visit_step += 1;
                *next_cell = Cell::Visited {
                    id,
                    visit_step: self.visit_step,
                };
                self.make_n_unique_steps(step_count);
                self.current_pos_x = prev_pos_x;
                self.current_pos_y = prev_pos_y;
            }
        }
    }
}

fn main() {
    const N: usize = 10;
    const M: usize = 10;
    const N_EXTENDED: usize = N + 2;
    const M_EXTENDED: usize = M + 2;
    let mut grid = Grid::<N, M, N_EXTENDED, M_EXTENDED>::new(
        [
            [Cell::Blocked; N],
            [
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(1),
                Cell::new(2),
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(3),
                Cell::new(4),
                Cell::Blocked,
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::new(5),
                Cell::new(6),
                Cell::new(7),
                Cell::new(8),
                Cell::new(9),
                Cell::new(10),
                Cell::new(11),
                Cell::new(12),
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::Blocked,
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(13),
                Cell::new(14),
                Cell::Blocked,
                Cell::Blocked,
                Cell::Blocked,
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::new(15),
                Cell::new(16),
                Cell::new(17),
                Cell::new(18),
                Cell::new(19),
                Cell::new(20),
                Cell::new(21),
                Cell::new(22),
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(23),
                Cell::new(24),
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(25),
                Cell::new(26),
                Cell::Blocked,
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::new(27),
                Cell::new(28),
                Cell::new(29),
                Cell::new(30),
                Cell::new(31),
                Cell::new(32),
                Cell::new(33),
                Cell::new(34),
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::new(35),
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(36),
                Cell::new(37),
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(38),
                Cell::Blocked,
            ],
            [
                Cell::Blocked,
                Cell::Blocked,
                Cell::new(39),
                Cell::new(40),
                Cell::new(41),
                Cell::new(42),
                Cell::new(43),
                Cell::new(44),
                Cell::Blocked,
                Cell::Blocked,
            ],
            [Cell::Blocked; N],
        ],
        (7, 1),
        [
            MoveDirection::Left,
            MoveDirection::Up,
            MoveDirection::Down,
            MoveDirection::Right,
        ],
    );
    grid.make_n_unique_steps(25);
    println!("{}", grid)
}
