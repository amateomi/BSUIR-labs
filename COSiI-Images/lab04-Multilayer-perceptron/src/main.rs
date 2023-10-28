use std::fs;

use eframe::{ egui, epaint::{ ColorImage, Color32 } };
use egui_extras::RetainedImage;
use image::EncodableLayout;
use rand::{ self, Rng };
use std::collections::HashMap;
use std::hash::Hash;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Multilayer perceptron",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::<Application>::default())
    )
}

const LEARNING_REPETITION_COUNT: usize = 100;

const SOURCE_IMAGES_COUNT: usize = 5;
const SOURCE_WIDTH: usize = 6;
const SOURCE_HEIGHT: usize = 6;

const INPUT_LAYER_SIZE: usize = SOURCE_WIDTH * SOURCE_HEIGHT;
const OUTPUT_LAYER_SIZE: usize = SOURCE_IMAGES_COUNT;
const INNER_LAYER_SIZE: usize = 4;

struct InputLayer([f32; INPUT_LAYER_SIZE]);

impl Hash for InputLayer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash_slice(self.0.as_bytes(), state)
    }
}

type Examples = HashMap<InputLayer, [f32; OUTPUT_LAYER_SIZE]>;

struct Application {
    source_images: [ColorImage; SOURCE_IMAGES_COUNT],
    test_images: Vec<ColorImage>,

    current_test_image_index: usize,

    v: [[f32; INNER_LAYER_SIZE]; INPUT_LAYER_SIZE],
    w: [[f32; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE],
    q: [f32; INNER_LAYER_SIZE],
    t: [f32; OUTPUT_LAYER_SIZE],

    examples: Examples,
}

impl Default for Application {
    fn default() -> Self {
        let mut file_content = fs::read_to_string("source.txt").expect("Valid source images path");
        file_content.retain(|char| !char.is_whitespace());

        let bit_to_color = |char: u8| {
            if char == b'1' { 255 } else { 0 }
        };
        let parse_file_format = |(count, image): (usize, &mut ColorImage)| {
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
        source_images.iter_mut().enumerate().for_each(parse_file_format);

        let mut file_content = fs::read_to_string("test.txt").expect("Valid test images path");
        file_content.retain(|char| !char.is_whitespace());

        let mut test_images =
            vec![
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED);
            file_content.len() / (SOURCE_WIDTH * SOURCE_HEIGHT)
        ];
        test_images.iter_mut().enumerate().for_each(parse_file_format);

        let mut rng = rand::thread_rng();

        let mut v = [[0.0f32; INNER_LAYER_SIZE]; INPUT_LAYER_SIZE];
        v.iter_mut().for_each(|row| { row.fill_with(|| { rng.gen_range(-1.0..1.0) }) });

        let mut w = [[0.0f32; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE];
        w.iter_mut().for_each(|row| { row.fill_with(|| { rng.gen_range(-1.0..1.0) }) });

        let mut q: [f32; INNER_LAYER_SIZE] = [0.0f32; INNER_LAYER_SIZE];
        q.fill_with(|| { rng.gen_range(-1.0..1.0) });

        let mut t: [f32; OUTPUT_LAYER_SIZE] = [0.0f32; OUTPUT_LAYER_SIZE];
        t.fill_with(|| { rng.gen_range(-1.0..1.0) });

        let mut examples = Examples::new();
        source_images
            .iter()
            .enumerate()
            .for_each(|(index, input)| {
                let mut output = [0.0f32; OUTPUT_LAYER_SIZE];
                output
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, item)| {
                        *item = if i == index { 1.0f32 } else { 0.0f32 };
                    });
                examples.insert(image_to_input_layer(&input), output);
            });
        Self {
            source_images,
            test_images,

            current_test_image_index: 0,

            v,
            w,
            q,
            t,

            examples,
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    for i in 0..self.source_images.len() {
                        RetainedImage::from_color_image(
                            format!("{i}"),
                            self.source_images[i].clone()
                        ).show_scaled(ui, 30.0);
                    }
                });
            });
        });
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
