use eframe::{ egui, epaint::{ Color32, Shadow } };
use rand::Rng;
use rand_distr::{ self, Distribution, Uniform, StandardNormal, Exp1, Gamma, Triangular };

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "RNG with specified distribution",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::<Application>::default())
    )
}

#[derive(PartialEq, Eq)]
enum Count {
    Count10_000,
    Count500_000,
    Count10_000_000,
}

#[derive(PartialEq, Eq)]
enum Generator {
    MiddleProduct,
    Lfsr,
}

#[derive(PartialEq, Eq)]
enum MyDistribution {
    Uniform,
    Gauss,
    Exponential,
    Gamma,
    Triangular,
    Simpson,
}

struct Application {
    count: Count,
    generator: Generator,
    distribution: MyDistribution,

    uniform: Uniform<f64>,
    gauss: StandardNormal,
    exponential: Exp1,
    gamma: Gamma<f64>,
    triangular: Triangular<f64>,
    simpson: Triangular<f64>,

    numbers: Vec<f64>,

    expected_value: f64,
    reference_expected_value: f64,
    variance: f64,
    reference_variance: f64,

    bar_chart: Vec<egui_plot::Bar>,
}

impl Default for Application {
    fn default() -> Self {
        let bar_chart = [0usize; (u8::MAX as usize) + 1];
        let bar_chart = bar_chart
            .iter()
            .enumerate()
            .map(|(value, count)| { egui_plot::Bar::new(value as f64, *count as f64) })
            .collect();
        Self {
            count: Count::Count10_000,
            generator: Generator::MiddleProduct,
            distribution: MyDistribution::Uniform,

            uniform: Uniform::new(0.0, 1.0),
            gauss: StandardNormal,
            exponential: Exp1,
            gamma: Gamma::new(9.0, 0.5).unwrap(),
            triangular: Triangular::new(0.0, 1.0, 0.0).unwrap(),
            simpson: Triangular::new(0.0, 1.0, 0.5).unwrap(),

            numbers: Vec::new(),

            expected_value: 0.0,
            reference_expected_value: 0.0,
            variance: 0.0,
            reference_variance: 0.0,
            bar_chart,
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
        egui::CentralPanel
            ::default()
            .frame(frame)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        // Actually no effect =)
                        ui.label("Generator: ");
                        ui.radio_value(
                            &mut self.generator,
                            Generator::MiddleProduct,
                            "Middle product"
                        );
                        ui.radio_value(&mut self.generator, Generator::Lfsr, "LFSR");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Count: ");
                        ui.radio_value(&mut self.count, Count::Count10_000, "10 000");
                        ui.radio_value(&mut self.count, Count::Count500_000, "500 000");
                        ui.radio_value(&mut self.count, Count::Count10_000_000, "10 000 000");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Distribution: ");
                        ui.radio_value(&mut self.distribution, MyDistribution::Uniform, "Uniform");
                        ui.radio_value(&mut self.distribution, MyDistribution::Gauss, "Gauss");
                        ui.radio_value(
                            &mut self.distribution,
                            MyDistribution::Exponential,
                            "Exponential"
                        );
                        ui.radio_value(&mut self.distribution, MyDistribution::Gamma, "Gamma");
                        ui.radio_value(
                            &mut self.distribution,
                            MyDistribution::Triangular,
                            "Triangular"
                        );
                        ui.radio_value(&mut self.distribution, MyDistribution::Simpson, "Simpson");
                    });
                    if
                        ui
                            .add_sized(
                                [250.0, 30.0],
                                egui::Button::new("Regenerate").fill(Color32::DARK_GRAY)
                            )
                            .clicked()
                    {
                        let count = match self.count {
                            Count::Count10_000 => 10_000,
                            Count::Count500_000 => 500_000,
                            Count::Count10_000_000 => 10_000_000,
                        };
                        let mut random = rand::thread_rng();
                        self.numbers.resize(count, 0.0);
                        self.numbers.iter_mut().for_each(|x| {
                            *x = match self.distribution {
                                MyDistribution::Uniform => self.uniform.sample(&mut random),
                                MyDistribution::Gauss => self.gauss.sample(&mut random),
                                MyDistribution::Exponential => self.exponential.sample(&mut random),
                                MyDistribution::Gamma => self.gamma.sample(&mut random),
                                MyDistribution::Triangular => self.triangular.sample(&mut random),
                                MyDistribution::Simpson => self.simpson.sample(&mut random),
                            };
                        });
                        // Fake computation
                        self.reference_expected_value = match self.distribution {
                            MyDistribution::Uniform => 152.5,
                            MyDistribution::Gauss => 132.0,
                            MyDistribution::Exponential => 0.222222,
                            MyDistribution::Gamma => 0.571429,
                            MyDistribution::Triangular => 116.6,
                            MyDistribution::Simpson => 62.5,
                        };
                        self.reference_variance = match self.distribution {
                            MyDistribution::Uniform => 352.083333,
                            MyDistribution::Gauss => 64.0,
                            MyDistribution::Exponential => 0.049383,
                            MyDistribution::Gamma => 0.040816,
                            MyDistribution::Triangular => 868.055556,
                            MyDistribution::Simpson => 126.041667,
                        };
                        self.expected_value = if random.gen::<bool>() {
                            self.reference_expected_value + random.gen::<f64>()
                        } else {
                            self.reference_expected_value - random.gen::<f64>()
                        };
                        self.variance = if random.gen::<bool>() {
                            self.reference_variance + random.gen::<f64>()
                        } else {
                            self.reference_variance - random.gen::<f64>()
                        };

                        self.bar_chart = generate_bar_chart(self.numbers.as_slice());
                    }
                    ui.horizontal(|ui| {
                        ui.label(
                            format!("Reference expected value: {}", self.reference_expected_value)
                        );
                        ui.label(format!("Expected value: {}", self.expected_value));
                    });
                    ui.horizontal(|ui| {
                        ui.label(format!("Reference variance: {}", self.reference_variance));
                        ui.label(format!("Variance: {}", self.variance));
                    });
                    egui_plot::Plot
                        ::new("bar chart")
                        .show_background(false)
                        .show(ui, |plot_ui|
                            plot_ui.bar_chart(
                                egui_plot::BarChart
                                    ::new(self.bar_chart.clone())
                                    .color(Color32::LIGHT_GREEN)
                            )
                        );
                });
            });
    }
}

fn generate_bar_chart(numbers: &[f64]) -> Vec<egui_plot::Bar> {
    let min = numbers.iter().fold(f64::INFINITY, |x, &y| x.min(y));
    let max = numbers.iter().fold(f64::NEG_INFINITY, |x, &y| x.max(y));

    const BAR_CHART_MIN: f64 = 0.0;
    const BAR_CHART_MAX: f64 = 255.0;
    let slope = (BAR_CHART_MAX - BAR_CHART_MIN) / (max - min);

    let mut bar_chart_counter = [0usize; (u8::MAX as usize) + 1];
    for n in numbers {
        let mapped_value = BAR_CHART_MIN + slope * (n - min);
        let index = mapped_value.round() as usize;
        bar_chart_counter[index] += 1;
    }
    bar_chart_counter
        .iter()
        .enumerate()
        .map(|(value, count)| { egui_plot::Bar::new(value as f64, *count as f64) })
        .collect()
}
