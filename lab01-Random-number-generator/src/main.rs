use std::{ collections::HashSet, hash::Hash };

use eframe::egui;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Random number generator",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::<Application>::default())
    )
}

struct Application {
    lehmer_random_number_generator: LehmerRandomNumberGenerator,
    random_numbers_sequence: Vec<u64>,
    random_numbers_bar_chart: Vec<egui_plot::Bar>,
}

impl Default for Application {
    fn default() -> Self {
        let mut lehmer_random_number_generator = LehmerRandomNumberGenerator::new(
            LehmerRandomNumberGenerator::DEFAULT_SEED,
            LehmerRandomNumberGenerator::DEFAULT_A,
            LehmerRandomNumberGenerator::DEFAULT_M
        );
        let random_numbers_sequence: Vec<u64> =
            lehmer_random_number_generator.get_n_random_numbers(10000);
        let random_numbers_bar_chart = generate_bar_chart(&random_numbers_sequence);
        Self {
            lehmer_random_number_generator,
            random_numbers_sequence,
            random_numbers_bar_chart,
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                egui_plot::Plot::new("random numbers").show(ui, |plot_ui| {
                    plot_ui.bar_chart(
                        egui_plot::BarChart::new(self.random_numbers_bar_chart.clone())
                    );
                });
            });
        });
    }
}

type RandomNumber = u64;
type RandomNumbers = Vec<RandomNumber>;

trait RandomNumberGenerator {
    fn clear_state(&mut self);

    fn get_next_random_number(&mut self) -> RandomNumber;

    fn get_n_random_numbers(&mut self, n: usize) -> RandomNumbers {
        (0..n).map(|_| { self.get_next_random_number() }).collect()
    }
}

fn generate_bar_chart(random_numbers: &RandomNumbers) -> Vec<egui_plot::Bar> {
    let mut 
    let mut random_number_counter = Vec::<u64>::new();
    random_number_counter.resize(random_numbers.iter().collect::<HashSet<_>>().len(), 0);
    for number in random_numbers {
        random_number_counter[*number as usize] += 1;
    }
    random_number_counter
        .iter()
        .enumerate()
        .map(|(x, y)| egui_plot::Bar::new(x as f64, *y as f64))
        .collect()
}

struct LehmerRandomNumberGenerator {
    seed: u64,
    a: u64,
    m: u64,
    prev: u64,
}

impl LehmerRandomNumberGenerator {
    const DEFAULT_SEED: u64 = 7;

    const DEFAULT_A: u64 = 75;
    const DEFAULT_M: u64 = (u16::MAX as u64) + 2;

    fn new(seed: u64, a: u64, m: u64) -> Self {
        Self { seed, a, m, prev: seed }
    }
}

impl RandomNumberGenerator for LehmerRandomNumberGenerator {
    fn clear_state(&mut self) {
        self.prev = self.seed;
    }

    fn get_next_random_number(&mut self) -> u64 {
        self.prev = (self.a * self.prev) % self.m;
        self.prev
    }
}
