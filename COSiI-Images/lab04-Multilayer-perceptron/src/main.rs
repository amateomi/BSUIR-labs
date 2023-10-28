use std::{ collections::HashMap, fs, hash::Hash };

use eframe::{ egui::{ self, TextureOptions }, epaint::{ Color32, ColorImage } };
use egui_extras::RetainedImage;
use image::EncodableLayout;
use itertools::izip;
use rand::{ self, Rng };

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Multilayer perceptron",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::<Application>::default())
    )
}

const SOURCE_IMAGES_COUNT: usize = 5;
const SOURCE_WIDTH: usize = 6;
const SOURCE_HEIGHT: usize = 6;

const INPUT_LAYER_SIZE: usize = SOURCE_WIDTH * SOURCE_HEIGHT;
const OUTPUT_LAYER_SIZE: usize = SOURCE_IMAGES_COUNT;
const INNER_LAYER_SIZE: usize = 4;

#[derive(PartialEq)]
struct InputLayer([f32; INPUT_LAYER_SIZE]);

impl Eq for InputLayer {}

impl Hash for InputLayer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash_slice(self.0.as_bytes(), state)
    }
}

type Examples = HashMap<InputLayer, [f32; OUTPUT_LAYER_SIZE]>;

struct Application {
    source_images: [ColorImage; SOURCE_IMAGES_COUNT],
    test_images: Vec<ColorImage>,

    was_test_image_change: bool,
    current_test_image_index: usize,
    current_test_image_output: [f32; OUTPUT_LAYER_SIZE],

    v: [[f32; INPUT_LAYER_SIZE]; INNER_LAYER_SIZE],
    w: [[f32; INNER_LAYER_SIZE]; OUTPUT_LAYER_SIZE],
    w_transposed: [[f32; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE],
    q: [f32; INNER_LAYER_SIZE],
    t: [f32; OUTPUT_LAYER_SIZE],

    alpha: f32,
    beta: f32,

    examples: Examples,
}

impl Default for Application {
    fn default() -> Self {
        let mut file_content = fs::read_to_string("source.txt").expect("Valid source images path");
        file_content.retain(|char| !char.is_whitespace());

        let bit_to_color = |char: u8| {
            if char == b'1' { 255 } else { 0 }
        };
        let parse_file_format = |(count, image): (usize, &mut ColorImage), file_content: &str| {
            for y in 0..SOURCE_HEIGHT {
                for x in 0..SOURCE_WIDTH {
                    let i = y * SOURCE_WIDTH + x;
                    image.pixels[i] = Color32::from_gray(
                        bit_to_color(
                            file_content.as_bytes()[count * SOURCE_WIDTH * SOURCE_HEIGHT + i]
                        )
                    );
                }
            }
        };

        let mut source_images: [ColorImage; OUTPUT_LAYER_SIZE] = [
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
        ];
        source_images
            .iter_mut()
            .enumerate()
            .for_each(|item| parse_file_format(item, file_content.as_str()));

        let mut file_content = fs::read_to_string("test.txt").expect("Valid test images path");
        file_content.retain(|char| !char.is_whitespace());

        let mut test_images =
            vec![
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED);
            file_content.len() / (SOURCE_WIDTH * SOURCE_HEIGHT)
        ];
        test_images
            .iter_mut()
            .enumerate()
            .for_each(|item| parse_file_format(item, file_content.as_str()));

        let mut rng = rand::thread_rng();

        let mut v = [[0.0f32; INPUT_LAYER_SIZE]; INNER_LAYER_SIZE];
        v.iter_mut().for_each(|row| row.fill_with(|| rng.gen_range(-1.0..1.0)));

        let mut w = [[0.0f32; INNER_LAYER_SIZE]; OUTPUT_LAYER_SIZE];
        w.iter_mut().for_each(|row| row.fill_with(|| rng.gen_range(-1.0..1.0)));

        let mut q: [f32; INNER_LAYER_SIZE] = [0.0f32; INNER_LAYER_SIZE];
        q.fill_with(|| rng.gen_range(-1.0..1.0));

        let mut t: [f32; OUTPUT_LAYER_SIZE] = [0.0f32; OUTPUT_LAYER_SIZE];
        t.fill_with(|| rng.gen_range(-1.0..1.0));

        let mut examples = Examples::new();
        source_images
            .iter()
            .enumerate()
            .for_each(|(index, input)| {
                let mut output = [0.0; OUTPUT_LAYER_SIZE];
                output
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item)| {
                        *item = if i == index { 1.0 } else { 0.0 };
                    });
                examples.insert(image_to_input_layer(input), output);
            });

        (Self {
            source_images,
            test_images,

            was_test_image_change: true,
            current_test_image_index: 0,
            current_test_image_output: [0.0; OUTPUT_LAYER_SIZE],

            v,
            w,
            w_transposed: [[0.0; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE],
            q,
            t,

            alpha: 0.6,
            beta: 0.6,

            examples,
        }).learn()
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.heading("Source images:");
                ui.horizontal(|ui| {
                    self.source_images
                        .iter()
                        .enumerate()
                        .for_each(|(i, image)| {
                            RetainedImage::from_color_image(
                                format!("source_image_{i}"),
                                image.clone()
                            )
                                .with_options(TextureOptions::NEAREST)
                                .show_size(ui, egui::Vec2::new(300.0, 300.0));
                        });
                });

                ui.horizontal(|ui| {
                    if ui.button("<-").clicked() && self.current_test_image_index > 0 {
                        self.current_test_image_index -= 1;
                        self.was_test_image_change = true;
                    }
                    if
                        ui.button("->").clicked() &&
                        self.current_test_image_index < self.test_images.len() - 1
                    {
                        self.current_test_image_index += 1;
                        self.was_test_image_change = true;
                    }
                });

                let current_test_image = &self.test_images[self.current_test_image_index];

                ui.label(format!("Current test image index: {}", self.current_test_image_index));
                RetainedImage::from_color_image(
                    format!("test_image_{}", self.current_test_image_index),
                    current_test_image.clone()
                )
                    .with_options(TextureOptions::NEAREST)
                    .show_size(ui, egui::Vec2::new(300.0, 300.0));

                self.current_test_image_output
                    .iter()
                    .enumerate()
                    .for_each(|(out, value)| {
                        ui.label(format!("out{out}: {value}"));
                    });
                if self.was_test_image_change {
                    self.current_test_image_output = compute_layer_output(
                        compute_layer_output(
                            image_to_input_layer(current_test_image).0.as_slice(),
                            &self.v,
                            self.q.as_slice()
                        ).as_slice(),
                        &self.w,
                        self.t.as_slice()
                    )
                        .try_into()
                        .unwrap();
                    self.was_test_image_change = false;
                }
            });
        });
    }
}

impl Application {
    fn learn(mut self) -> Self {
        'learn_loop: loop {
            for (input, target_output) in self.examples.iter() {
                transpose(self.w, &mut self.w_transposed);

                let g = compute_layer_output(input.0.as_slice(), &self.v, self.q.as_slice());
                let y = compute_layer_output(g.as_slice(), &self.w, self.t.as_slice());

                let d = compute_output_neuron_errors(y.as_slice(), target_output);
                let e = compute_inner_neuron_errors(y.as_slice(), d.as_slice(), &self.w_transposed);

                izip!(self.w.iter_mut(), y.clone(), d.clone()).for_each(|(w, y, d)| {
                    w.iter_mut()
                        .zip(g.clone())
                        .for_each(|(w, g)| {
                            *w += self.alpha * sigmoid_derivative(y) * d * g;
                        })
                });
                izip!(self.t.iter_mut(), y.clone(), d.clone()).for_each(|(t, y, d)| {
                    *t += self.alpha * sigmoid_derivative(y) * d;
                });
                izip!(self.v.iter_mut(), g.clone(), e.clone()).for_each(|(v, g, e)| {
                    v.iter_mut()
                        .zip(input.0)
                        .for_each(|(v, x)| {
                            *v += self.beta * sigmoid_derivative(g) * e * x;
                        })
                });
                izip!(self.q.iter_mut(), g.clone(), e.clone()).for_each(|(q, g, e)| {
                    *q += self.beta * sigmoid_derivative(g) * e;
                });

                if d.iter().all(|x| *x < 0.01) {
                    break 'learn_loop;
                }
            }
        }
        transpose(self.w, &mut self.w_transposed);

        self
    }
}

fn transpose<const U: usize, const V: usize>(input: [[f32; U]; V], output: &mut [[f32; V]; U]) {
    for i in 0..V {
        for j in 0..U {
            output[j][i] = input[i][j];
        }
    }
}

fn image_to_input_layer(image: &ColorImage) -> InputLayer {
    InputLayer(
        image.pixels
            .iter()
            .map(|color| {
                match *color {
                    Color32::BLACK => 1.0f32,
                    Color32::WHITE => 0.0f32,
                    _ => panic!("Unsupported color"),
                }
            })
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap()
    )
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

/// Sk
fn compute_weighted_sum(input: &[f32], links: &[f32], threshold: f32) -> f32 {
    links
        .iter()
        .zip(input)
        .map(|(w, x)| w * x)
        .sum::<f32>() + threshold
}

fn compute_layer_output<T>(input: &[f32], weights: &[T], thresholds: &[f32]) -> Vec<f32>
    where T: AsRef<[f32]>
{
    weights
        .iter()
        .zip(thresholds)
        .map(|(links, threshold)| sigmoid(compute_weighted_sum(input, links.as_ref(), *threshold)))
        .collect()
}

fn compute_output_neuron_errors(actual: &[f32], target: &[f32]) -> Vec<f32> {
    actual
        .iter()
        .zip(target)
        .map(|(y, target_y)| target_y - y)
        .collect()
}

fn compute_inner_neuron_errors<T>(output: &[f32], output_error: &[f32], weights: &[T]) -> Vec<f32>
    where T: AsRef<[f32]>
{
    weights
        .iter()
        .map(|links| {
            izip!(output, output_error, links.as_ref())
                .map(|(y, d, w)| d * sigmoid_derivative(*y) * w)
                .sum()
        })
        .collect()
}
