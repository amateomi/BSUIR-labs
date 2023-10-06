use eframe::egui;
use nistrs::prelude::*;
use rand::Rng;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Random number generator",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::<Application>::default())
    )
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    LehmerFirst,
    LehmerSecond,
    LehmerThird,
}

struct Application {
    current_tab: Tab,

    lehmer: ApplicationGeneratorContent<LehmerRandomNumberGenerator>,

    lehmer_seed_field: String,
    lehmer_a_field: String,
    lehmer_m_field: String,
}

impl Default for Application {
    fn default() -> Self {
        Self {
            current_tab: Tab::LehmerFirst,
            lehmer: ApplicationGeneratorContent::new(
                LehmerRandomNumberGenerator::new(
                    LehmerRandomNumberGenerator::SEED,
                    LehmerRandomNumberGenerator::A,
                    LehmerRandomNumberGenerator::M
                )
            ),
            lehmer_seed_field: String::default(),
            lehmer_a_field: String::default(),
            lehmer_m_field: String::default(),
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    self.current_tab = if ui.button("Lehmer 10 000 samples").clicked() {
                        Tab::LehmerFirst
                    } else if ui.button("Lehmer 500 000 samples").clicked() {
                        Tab::LehmerSecond
                    } else if ui.button("Lehmer 100 000 000 samples").clicked() {
                        Tab::LehmerThird
                    } else {
                        self.current_tab
                    };
                });

                match self.current_tab {
                    Tab::LehmerFirst | Tab::LehmerSecond | Tab::LehmerThird => {
                        ui.horizontal(|ui| {
                            ui.label("seed: ");
                            if ui.text_edit_singleline(&mut self.lehmer_seed_field).changed() {
                                if let Ok(seed) = self.lehmer_a_field.parse::<u64>() {
                                    self.lehmer.generator.seed = seed;
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("a:     ");
                            if ui.text_edit_singleline(&mut self.lehmer_a_field).changed() {
                                if let Ok(a) = self.lehmer_a_field.parse::<u64>() {
                                    self.lehmer.generator.a = a;
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("m:    ");
                            if ui.text_edit_singleline(&mut self.lehmer_m_field).changed() {
                                if let Ok(m) = self.lehmer_m_field.parse::<u64>() {
                                    self.lehmer.generator.m = m;
                                }
                            }
                        });
                    }
                }

                let generator = match self.current_tab {
                    Tab::LehmerFirst | Tab::LehmerSecond | Tab::LehmerThird =>
                        &mut self.lehmer.generator,
                };
                let tab_content = match self.current_tab {
                    Tab::LehmerFirst => &mut self.lehmer.first_tab,
                    Tab::LehmerSecond => &mut self.lehmer.second_tab,
                    Tab::LehmerThird => &mut self.lehmer.third_tab,
                };

                if ui.button("recompute tests").clicked() {
                    let mut samples = Vec::<u64>::new();
                    samples.resize(tab_content.samples_count, 0);
                    generator.clear_state().generate(samples.as_mut_slice());
                    *tab_content = ApplicationTabContent::new(samples);
                }

                ui.horizontal(|ui| {
                    tab_content.test_results
                        .iter()
                        .enumerate()
                        .for_each(|(i, res)| {
                            ui.label(format!("Test{}: {}", i + 1, res));
                        })
                });
                ui.label(format!("Expected value: {}", tab_content.expected_value));
                ui.label(format!("Variance: {}", tab_content.variance));
                egui_plot::Plot
                    ::new("bar chart")
                    .show(ui, |plot_ui|
                        plot_ui.bar_chart(egui_plot::BarChart::new(tab_content.bar_chart.clone()))
                    );
            });
        });
    }
}

const SAMPLES_FIRST_COUNT: usize = 10_000;
const SAMPLES_SECOND_COUNT: usize = 500_000;
const SAMPLES_THIRD_COUNT: usize = 100_000_000;

struct ApplicationGeneratorContent<R> where R: RandomNumberGenerator {
    generator: R,
    first_tab: ApplicationTabContent,
    second_tab: ApplicationTabContent,
    third_tab: ApplicationTabContent,
}

impl<R> ApplicationGeneratorContent<R> where R: RandomNumberGenerator {
    fn new(mut generator: R) -> Self {
        let mut first_samples = Vec::<u64>::new();
        let mut second_samples = Vec::<u64>::new();
        let mut third_samples = Vec::<u64>::new();

        first_samples.resize(SAMPLES_FIRST_COUNT, 0);
        second_samples.resize(SAMPLES_SECOND_COUNT, 0);
        third_samples.resize(SAMPLES_THIRD_COUNT, 0);

        generator
            .generate(first_samples.as_mut_slice())
            .clear_state()
            .generate(second_samples.as_mut_slice())
            .clear_state()
            .generate(third_samples.as_mut_slice())
            .clear_state();

        Self {
            generator,
            first_tab: ApplicationTabContent::new(first_samples),
            second_tab: ApplicationTabContent::new(second_samples),
            third_tab: ApplicationTabContent::new(third_samples),
        }
    }
}

struct ApplicationTabContent {
    samples_count: usize,
    test_results: [bool; 8],
    expected_value: f64,
    variance: f64,
    bar_chart: Vec<egui_plot::Bar>,
}

impl ApplicationTabContent {
    fn new(samples: Vec<u64>) -> Self {
        let middle = ((samples.iter().max().unwrap() - samples.iter().min().unwrap()) as f64) / 2.0;
        let bits = BitsData::from_text(
            samples
                .iter()
                .map(|x| if (*x as f64) < middle { '0' } else { '1' })
                .collect::<String>()
        );
        Self {
            samples_count: samples.len(),
            test_results: [
                frequency_test(&bits).0,
                block_frequency_test(&bits, 8).is_ok_and(|(x, _)| x),
                runs_test(&bits).0,
                longest_run_of_ones_test(&bits).is_ok_and(|(x, _)| x),
                rank_test(&bits).is_ok_and(|(x, _)| x),
                fft_test(&bits).0,
                non_overlapping_template_test(&bits, 8).is_ok_and(|v| v.iter().all(|(x, _)| *x)),
                overlapping_template_test(&bits, 8).0,
            ],
            expected_value: compute_expected_value(&samples),
            variance: compute_variance(&samples),
            bar_chart: generate_bar_chart(&samples),
        }
    }
}

fn compute_expected_value(numbers: &[u64]) -> f64 {
    let sum = numbers.iter().sum::<u64>() as f64;
    let count = numbers.len() as f64;
    sum / count
}

fn compute_variance(numbers: &[u64]) -> f64 {
    let expected_value = compute_expected_value(numbers);
    let sum = numbers
        .iter()
        .map(|x| {
            let diff = (*x as f64) - expected_value;
            diff * diff
        })
        .sum::<f64>();
    let count = numbers.len() as f64;
    sum / count
}

fn generate_bar_chart(numbers: &[u64]) -> Vec<egui_plot::Bar> {
    let min = numbers.iter().min().unwrap();
    let max = numbers.iter().max().unwrap();

    const BAR_CHART_MIN: f64 = 0.0;
    const BAR_CHART_MAX: f64 = 255.0;
    let slope = (BAR_CHART_MAX - BAR_CHART_MIN) / ((max - min) as f64);

    let mut bar_chart_counter = [0usize; (u8::MAX as usize) + 1];
    for n in numbers {
        let mapped_value = BAR_CHART_MIN + slope * ((n - min) as f64);
        let index = mapped_value as usize;
        bar_chart_counter[index] += 1;
    }
    bar_chart_counter
        .iter()
        .enumerate()
        .map(|(value, count)| { egui_plot::Bar::new(value as f64, *count as f64) })
        .collect()
}

trait RandomNumberGenerator {
    fn clear_state(&mut self) -> &mut Self;

    fn get_next_random_number(&mut self) -> u64;

    fn generate(&mut self, numbers: &mut [u64]) -> &mut Self {
        numbers.iter_mut().for_each(|n| {
            *n = self.get_next_random_number();
        });
        self
    }
}

struct LehmerRandomNumberGenerator {
    seed: u64,
    a: u64,
    m: u64,
    prev: u64,
}

impl LehmerRandomNumberGenerator {
    const SEED: u64 = 7;
    const A: u64 = 75;
    const M: u64 = (u16::MAX as u64) + 2;

    fn new(seed: u64, a: u64, m: u64) -> Self {
        Self { seed, a, m, prev: seed }
    }
}

impl RandomNumberGenerator for LehmerRandomNumberGenerator {
    fn clear_state(&mut self) -> &mut Self {
        self.prev = self.seed;
        self
    }

    fn get_next_random_number(&mut self) -> u64 {
        self.prev = (self.a * self.prev) % self.m;
        self.prev
    }
}
