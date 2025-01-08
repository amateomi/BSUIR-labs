use itertools::Itertools;
use std::array;
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
    UpRight,
    Right,
    DownRight,
    Down,
    DownLeft,
    Left,
    UpLeft,
}

impl Display for MoveDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        match self {
            MoveDirection::Up => write!(f, "[Up]"),
            MoveDirection::UpRight => write!(f, "[Up-Right]"),
            MoveDirection::Right => write!(f, "[Right]"),
            MoveDirection::DownRight => write!(f, "[Down-Right]"),
            MoveDirection::Down => write!(f, "[Down]"),
            MoveDirection::DownLeft => write!(f, "[Down-Left]"),
            MoveDirection::Left => write!(f, "[Left]"),
            MoveDirection::UpLeft => write!(f, "[Up-Left]"),
        }
    }
}

type BasicMoveLogic = [MoveDirection; 4];
type AdvancedMoveLogic = [MoveDirection; 8];

#[derive(Clone, Copy)]
enum MoveLogic {
    BasicMoveLogic { move_logic: BasicMoveLogic },
    AdvancedMoveLogic { move_logic: AdvancedMoveLogic },
}

impl Display for MoveLogic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result {
        match self {
            MoveLogic::BasicMoveLogic { move_logic } => write!(
                f,
                "basic move logic: {}->{}->{}->{}",
                move_logic[0], move_logic[1], move_logic[2], move_logic[3]
            ),

            MoveLogic::AdvancedMoveLogic { move_logic } => write!(
                f,
                "advanced move logic: {}->{}->{}->{}->{}->{}->{}->{}",
                move_logic[0],
                move_logic[1],
                move_logic[2],
                move_logic[3],
                move_logic[4],
                move_logic[5],
                move_logic[6],
                move_logic[7]
            ),
        }
    }
}

struct Grid<const N: usize, const M: usize, const N_EXTENDED: usize, const M_EXTENDED: usize> {
    grid: [[Cell; N_EXTENDED]; M_EXTENDED],
    current_pos_x: usize,
    current_pos_y: usize,
    move_logic: MoveLogic,
    visit_step: u32,
    is_path_found: bool,
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
        writeln!(f, "{}", self.move_logic)?;
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
            let move_logic = match move_logic {
                MoveLogic::BasicMoveLogic { move_logic } => move_logic.as_slice(),
                MoveLogic::AdvancedMoveLogic { move_logic } => move_logic.as_slice(),
            };
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
            is_path_found: false,
        }
    }

    fn find_path(&mut self, end_pos: (usize, usize)) {
        let move_logic_copy = self.move_logic;
        let move_logic = match &move_logic_copy {
            MoveLogic::BasicMoveLogic { move_logic } => move_logic.as_slice(),
            MoveLogic::AdvancedMoveLogic { move_logic } => move_logic.as_slice(),
        };
        for move_direction in move_logic {
            if self.is_path_found {
                return;
            }
            let (end_pos_y, end_pos_x) = end_pos;
            if self.current_pos_x == (end_pos_x + 1) && self.current_pos_y == (end_pos_y + 1) {
                self.is_path_found = true;
                return;
            }
            let (next_pos_y, next_pos_x) = match self.move_logic {
                MoveLogic::BasicMoveLogic { .. } => match *move_direction {
                    MoveDirection::Up => (self.current_pos_y - 1, self.current_pos_x),
                    MoveDirection::Right => (self.current_pos_y, self.current_pos_x + 1),
                    MoveDirection::Down => (self.current_pos_y + 1, self.current_pos_x),
                    MoveDirection::Left => (self.current_pos_y, self.current_pos_x - 1),
                    _ => panic!(
                        "{} direction is invalid for basic move logic",
                        *move_direction
                    ),
                },
                MoveLogic::AdvancedMoveLogic { .. } => match *move_direction {
                    MoveDirection::Up => (self.current_pos_y - 1, self.current_pos_x),
                    MoveDirection::UpRight => (self.current_pos_y - 1, self.current_pos_x + 1),
                    MoveDirection::Right => (self.current_pos_y, self.current_pos_x + 1),
                    MoveDirection::DownRight => (self.current_pos_y + 1, self.current_pos_x + 1),
                    MoveDirection::Down => (self.current_pos_y + 1, self.current_pos_x),
                    MoveDirection::DownLeft => (self.current_pos_y + 1, self.current_pos_x - 1),
                    MoveDirection::Left => (self.current_pos_y, self.current_pos_x - 1),
                    MoveDirection::UpLeft => (self.current_pos_y - 1, self.current_pos_x - 1),
                },
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
                self.find_path(end_pos);
                self.current_pos_x = prev_pos_x;
                self.current_pos_y = prev_pos_y;
            }
        }
    }
}

fn form_grid_and_find_path(move_logic: &[MoveDirection]) -> u32 {
    const N: usize = 5;
    const M: usize = 5;
    const N_EXTENDED: usize = N + 2;
    const M_EXTENDED: usize = M + 2;

    let input_grid = [
        array::from_fn(|i| Cell::new(i as u32)),
        array::from_fn(|i| Cell::new((i + N) as u32)),
        [
            Cell::Blocked,
            Cell::new(10),
            Cell::Blocked,
            Cell::Blocked,
            Cell::new(11),
        ],
        array::from_fn(|i| Cell::new((i + N + 2) as u32)),
        array::from_fn(|i| Cell::new((i + 2 * N + 2) as u32)),
    ];
    let start_pos = (0, 0);
    let mut grid = match move_logic.len() {
        4 => Grid::<N, M, N_EXTENDED, M_EXTENDED>::new(
            input_grid,
            start_pos,
            MoveLogic::BasicMoveLogic {
                move_logic: array::from_fn(|i| move_logic[i]),
            },
        ),
        8 => Grid::<N, M, N_EXTENDED, M_EXTENDED>::new(
            input_grid,
            start_pos,
            MoveLogic::AdvancedMoveLogic {
                move_logic: array::from_fn(|i| move_logic[i]),
            },
        ),
        _ => panic!("Invalid move logic slice length: {}", move_logic.len()),
    };
    grid.find_path((4, 4));
    grid.visit_step
}

fn main() {
    {
        let basic_move_logic = [
            MoveDirection::Up,
            MoveDirection::Right,
            MoveDirection::Down,
            MoveDirection::Left,
        ];
        let mut best_basic_permutations = Vec::<(BasicMoveLogic, u32)>::default();
        for basic_move_logic_permutation in
            basic_move_logic.iter().permutations(basic_move_logic.len())
        {
            let basic_move_logic: BasicMoveLogic =
                array::from_fn(|i| *basic_move_logic_permutation[i]);
            let steps = form_grid_and_find_path(basic_move_logic.as_slice());
            if best_basic_permutations.is_empty() || best_basic_permutations[0].1 == steps {
                best_basic_permutations.push((basic_move_logic, steps));
            } else if best_basic_permutations[0].1 > steps {
                best_basic_permutations.clear();
                best_basic_permutations.push((basic_move_logic, steps));
            }
        }
        for (move_logic, steps) in best_basic_permutations {
            println!(
                "{}->{}->{}->{}: {} steps",
                move_logic[0], move_logic[1], move_logic[2], move_logic[3], steps
            );
        }
    }
    {
        let advanced_move_logic = [
            MoveDirection::Up,
            MoveDirection::UpRight,
            MoveDirection::Right,
            MoveDirection::DownRight,
            MoveDirection::Down,
            MoveDirection::DownLeft,
            MoveDirection::Left,
            MoveDirection::UpLeft,
        ];
        let mut best_advanced_permutations = Vec::<(AdvancedMoveLogic, u32)>::default();
        for advanced_move_logic_permutation in advanced_move_logic
            .iter()
            .permutations(advanced_move_logic.len())
        {
            let advanced_move_logic: AdvancedMoveLogic =
                array::from_fn(|i| *advanced_move_logic_permutation[i]);
            let steps = form_grid_and_find_path(advanced_move_logic.as_slice());
            if best_advanced_permutations.is_empty() || best_advanced_permutations[0].1 == steps {
                best_advanced_permutations.push((advanced_move_logic, steps));
            } else if best_advanced_permutations[0].1 > steps {
                best_advanced_permutations.clear();
                best_advanced_permutations.push((advanced_move_logic, steps));
            }
        }
        for (move_logic, steps) in best_advanced_permutations {
            println!(
                "{}->{}->{}->{}->{}->{}->{}->{}: {} steps",
                move_logic[0],
                move_logic[1],
                move_logic[2],
                move_logic[3],
                move_logic[4],
                move_logic[5],
                move_logic[6],
                move_logic[7],
                steps
            );
        }
    }
}
