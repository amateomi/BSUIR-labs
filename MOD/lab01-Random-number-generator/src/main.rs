use eframe::{ egui, epaint::{ Color32, Shadow } };
use nistrs::prelude::*;

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
    MiddleProductFirst,
    MiddleProductSecond,
    MiddleProductThird,
    LFSRFirst,
    LFSRSecond,
    LFSRThird,
}

struct Application {
    current_tab: Tab,

    lehmer_seed_field: String,
    lehmer_a_field: String,
    lehmer_m_field: String,

    lehmer: ApplicationGeneratorContent<LehmerRandomNumberGenerator>,
    middle_product: ApplicationGeneratorContent<MiddleProductRandomNumberGenerator>,
    lfsr: ApplicationGeneratorContent<LFSRRandomNumberGenerator>,
}

impl Default for Application {
    fn default() -> Self {
        Self {
            current_tab: Tab::LehmerFirst,
            lehmer_seed_field: String::default(),
            lehmer_a_field: String::default(),
            lehmer_m_field: String::default(),
            lehmer: ApplicationGeneratorContent::new(
                LehmerRandomNumberGenerator::new(
                    LehmerRandomNumberGenerator::SEED,
                    LehmerRandomNumberGenerator::A,
                    LehmerRandomNumberGenerator::M
                )
            ),
            middle_product: ApplicationGeneratorContent::new(
                MiddleProductRandomNumberGenerator::new(
                    MiddleProductRandomNumberGenerator::R0,
                    MiddleProductRandomNumberGenerator::R1
                )
            ),
            lfsr: ApplicationGeneratorContent::new(
                LFSRRandomNumberGenerator::new(LFSRRandomNumberGenerator::SEED)
            ),
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(2.0);
        let frame = egui::containers::Frame {
            inner_margin: egui::Margin::default(),
            outer_margin: egui::Margin::default(),
            rounding: egui::Rounding { nw: 1.0, ne: 1.0, se: 1.0, sw: 1.0 },
            fill: Color32::WHITE,
            stroke: egui::Stroke::new(0.0, Color32::GOLD),
            shadow: Shadow::small_light(),
        };
        let tab_content = match self.current_tab {
            Tab::LehmerFirst => &mut self.lehmer.first_tab,
            Tab::LehmerSecond => &mut self.lehmer.second_tab,
            Tab::LehmerThird => &mut self.lehmer.third_tab,
            Tab::MiddleProductFirst => &mut self.middle_product.first_tab,
            Tab::MiddleProductSecond => &mut self.middle_product.second_tab,
            Tab::MiddleProductThird => &mut self.middle_product.third_tab,
            Tab::LFSRFirst => &mut self.lfsr.first_tab,
            Tab::LFSRSecond => &mut self.lfsr.second_tab,
            Tab::LFSRThird => &mut self.lfsr.third_tab,
        };
        egui::CentralPanel
            ::default()
            .frame(frame)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            self.current_tab = if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Lehmer 10 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LehmerFirst
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Lehmer 500 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LehmerSecond
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Lehmer 100 000 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LehmerThird
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Middle product 10 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::MiddleProductFirst
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Middle product 500 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::MiddleProductSecond
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("Middle product 100 000 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::MiddleProductThird
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("LFSR 10 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LFSRFirst
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("LFSR 500 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LFSRSecond
                            } else if
                                ui
                                    .add_sized(
                                        [250.0, 30.0],
                                        egui::Button
                                            ::new("LFSR 100 000 000 samples")
                                            .fill(Color32::LIGHT_GREEN)
                                    )
                                    .clicked()
                            {
                                Tab::LFSRThird
                            } else {
                                self.current_tab
                            };
                        });
                        egui_plot::Plot
                            ::new("bar chart")
                            .show_background(false)
                            .show(ui, |plot_ui|
                                plot_ui.bar_chart(
                                    egui_plot::BarChart
                                        ::new(tab_content.bar_chart.clone())
                                        .color(Color32::LIGHT_GREEN)
                                )
                            );
                    });

                    if
                        ui
                            .add_sized(
                                [250.0, 30.0],
                                egui::Button::new("Recompute tests").fill(Color32::GREEN)
                            )
                            .clicked()
                    {
                        let generator: &mut dyn RandomNumberGenerator = match self.current_tab {
                            Tab::LehmerFirst | Tab::LehmerSecond | Tab::LehmerThird =>
                                self.lehmer.generator.as_mut(),
                            | Tab::MiddleProductFirst
                            | Tab::MiddleProductSecond
                            | Tab::MiddleProductThird => self.middle_product.generator.as_mut(),
                            Tab::LFSRFirst | Tab::LFSRSecond | Tab::LFSRThird =>
                                self.lfsr.generator.as_mut(),
                        };

                        let mut samples = Vec::<u64>::new();
                        samples.resize(tab_content.samples_count, 0);
                        generator.clear_state();
                        generator.generate(samples.as_mut_slice());
                        *tab_content = ApplicationTabContent::new(samples);
                    }

                    ui.horizontal(|ui| {
                        tab_content.test_results
                            .iter()
                            .enumerate()
                            .for_each(|(i, res)| {
                                ui.label(format!("Test {}: {}", i + 1, res));
                            })
                    });

                    ui.horizontal(|ui| {
                        ui.label(format!("Expected value: {}", tab_content.expected_value));
                        ui.label(
                            format!(
                                "Reference expected value: {}",
                                tab_content.reference_expected_value
                            )
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label(format!("Variance: {}", tab_content.variance));
                        ui.label(format!("Reference variance: {}", tab_content.reference_variance));
                    });

                    match self.current_tab {
                        Tab::LehmerFirst | Tab::LehmerSecond | Tab::LehmerThird => {
                            ui.label(" Lehmer parameters:");
                            ui.horizontal(|ui| {
                                ui.label(" seed: ");
                                if ui.text_edit_singleline(&mut self.lehmer_seed_field).changed() {
                                    if let Ok(seed) = self.lehmer_a_field.parse::<u64>() {
                                        self.lehmer.generator.seed = seed;
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label(" a:        ");
                                if ui.text_edit_singleline(&mut self.lehmer_a_field).changed() {
                                    if let Ok(a) = self.lehmer_a_field.parse::<u64>() {
                                        self.lehmer.generator.a = a;
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label(" m:      ");
                                if ui.text_edit_singleline(&mut self.lehmer_m_field).changed() {
                                    if let Ok(m) = self.lehmer_m_field.parse::<u64>() {
                                        self.lehmer.generator.m = m;
                                    }
                                }
                            });
                        }
                        _ => {}
                    }
                });
            });
    }
}

const SAMPLES_FIRST_COUNT: usize = 10_000;
const SAMPLES_SECOND_COUNT: usize = 500_000;
const SAMPLES_THIRD_COUNT: usize = 100_000_000;

struct ApplicationGeneratorContent<R> where R: RandomNumberGenerator {
    generator: Box<R>,
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

        generator.generate(first_samples.as_mut_slice());
        generator.clear_state();

        generator.generate(second_samples.as_mut_slice());
        generator.clear_state();

        generator.generate(third_samples.as_mut_slice());
        generator.clear_state();

        Self {
            generator: Box::new(generator),
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
    reference_expected_value: f64,
    variance: f64,
    reference_variance: f64,
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
            reference_expected_value: compute_reference_expected_value(),
            variance: compute_variance(&samples),
            reference_variance: compute_reference_variance(&samples),
            bar_chart: generate_bar_chart(&samples),
        }
    }
}

fn compute_expected_value(numbers: &[u64]) -> f64 {
    let sum = numbers
        .iter()
        .map(|x| *x as u128)
        .sum::<u128>() as f64;
    let count = numbers.len() as f64;
    sum / count
}

fn compute_reference_expected_value() -> f64 {
    (u64::MAX as f64) / 2.0
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

fn compute_reference_variance(numbers: &[u64]) -> f64 {
    let expected_value = compute_reference_expected_value();
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
        let index = mapped_value.round() as usize;
        bar_chart_counter[index] += 1;
    }
    bar_chart_counter
        .iter()
        .enumerate()
        .map(|(value, count)| { egui_plot::Bar::new(value as f64, *count as f64) })
        .collect()
}

trait RandomNumberGenerator {
    fn clear_state(&mut self);

    fn get_next_random_number(&mut self) -> u64;

    fn generate(&mut self, numbers: &mut [u64]) {
        numbers.iter_mut().for_each(|n| {
            *n = self.get_next_random_number();
        });
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
    const A: u64 = 16_666;
    const M: u64 = 999999999767;

    fn new(seed: u64, a: u64, m: u64) -> Self {
        Self { seed, a, m, prev: seed }
    }
}

impl RandomNumberGenerator for LehmerRandomNumberGenerator {
    fn clear_state(&mut self) {
        self.prev = self.seed;
    }

    fn get_next_random_number(&mut self) -> u64 {
        self.prev = self.a.overflowing_mul(self.prev).0 % self.m;
        self.prev
    }
}

struct MiddleProductRandomNumberGenerator {
    r0: u64,
    r1: u64,
    multiplier: u64,
    random_number: u64,
}

impl MiddleProductRandomNumberGenerator {
    const R0: u64 = 0xffeeddccbbaa9988;
    const R1: u64 = 0x1122334455667788;

    fn new(r0: u64, r1: u64) -> Self {
        Self { r0, r1, multiplier: r0, random_number: r1 }
    }
}

impl RandomNumberGenerator for MiddleProductRandomNumberGenerator {
    fn clear_state(&mut self) {
        self.multiplier = self.r0;
        self.random_number = self.r1;
    }

    fn get_next_random_number(&mut self) -> u64 {
        let multiply_result = (self.multiplier as u128) * (self.random_number as u128);
        self.multiplier = self.random_number;
        self.random_number = ((multiply_result & 0x00000000_ffffffff_ffffffff_00000000) >>
            32) as u64;
        self.random_number
    }
}

struct LFSRRandomNumberGenerator {
    seed: u64,
    shift_register: u64,
}

impl LFSRRandomNumberGenerator {
    const SEED: u64 = 0x0f0f0f0f_0f0f0f0f;

    fn new(seed: u64) -> Self {
        Self {
            seed,
            shift_register: seed,
        }
    }
}

impl RandomNumberGenerator for LFSRRandomNumberGenerator {
    fn clear_state(&mut self) {
        self.shift_register = self.seed;
    }

    fn get_next_random_number(&mut self) -> u64 {
        let mut result = 0;
        for i in 0..64 {
            self.shift_register =
                ((((self.shift_register >> 63) ^
                    (self.shift_register >> 62) ^
                    (self.shift_register >> 60) ^
                    (self.shift_register >> 59) ^
                    self.shift_register) &
                    1) <<
                    63) |
                (self.shift_register >> 1);
            let bit = self.shift_register & 1;
            result |= bit << i;
        }
        result
    }
}
