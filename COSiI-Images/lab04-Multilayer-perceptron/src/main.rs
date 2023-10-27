use std::fs;

use eframe::{ egui, epaint::{ ColorImage, Color32 } };
use egui_extras::RetainedImage;
use rand::{ self, Rng };

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
struct Application {
    source_images: [ColorImage; SOURCE_IMAGES_COUNT],
    test_images: Vec<ColorImage>,

    current_test_image_index: usize,

    v: [[f32; INNER_LAYER_SIZE]; INPUT_LAYER_SIZE],
    w: [[f32; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE],
    q: [f32; INNER_LAYER_SIZE],
    t: [f32; OUTPUT_LAYER_SIZE],
}

impl Default for Application {
    fn default() -> Self {
        let mut file_content = fs::read_to_string("source.txt").expect("Valid source images path");
        file_content.retain(|char| !char.is_whitespace());

        let bit_to_color = |char: u8| {
            if char == b'1' { 255 } else { 0 }
        };

        let mut source_images: [ColorImage; 5] = [
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
            ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED),
        ];
        (0..source_images.len()).for_each(|count| {
            dbg!(count);
            for y in 0..SOURCE_HEIGHT {
                for x in 0..SOURCE_WIDTH {
                    let i = y * SOURCE_WIDTH + x;
                    source_images[count].pixels[i] = Color32::from_gray(
                        bit_to_color(
                            file_content.as_bytes()[count * SOURCE_WIDTH * SOURCE_HEIGHT + i]
                        )
                    );
                }
            }
        });

        let mut file_content = fs::read_to_string("test.txt").expect("Valid test images path");
        file_content.retain(|char| !char.is_whitespace());

        let mut test_images = Vec::<ColorImage>::new();
        (0..file_content.len() / (SOURCE_WIDTH * SOURCE_HEIGHT)).for_each(|_| {
            let mut image = ColorImage::new([SOURCE_WIDTH, SOURCE_HEIGHT], Color32::RED);
            for y in 0..SOURCE_HEIGHT {
                for x in 0..SOURCE_WIDTH {
                    let i = y * SOURCE_WIDTH + x;
                    image.pixels[i] = Color32::from_gray(bit_to_color(file_content.as_bytes()[i]));
                }
            }
            test_images.push(image);
        });

        let mut rng = rand::thread_rng();

        let mut v = [[0.0f32; INNER_LAYER_SIZE]; INPUT_LAYER_SIZE];
        v.iter_mut().for_each(|row| { row.fill_with(|| { rng.gen_range(0.0..1.0) }) });

        let mut w = [[0.0f32; OUTPUT_LAYER_SIZE]; INNER_LAYER_SIZE];
        w.iter_mut().for_each(|row| { row.fill_with(|| { rng.gen_range(0.0..1.0) }) });

        let mut q: [f32; INNER_LAYER_SIZE] = [0.0f32; INNER_LAYER_SIZE];
        q.fill_with(|| { rng.gen_range(0.0..1.0) });

        let mut t: [f32; OUTPUT_LAYER_SIZE] = [0.0f32; OUTPUT_LAYER_SIZE];
        t.fill_with(|| { rng.gen_range(0.0..1.0) });

        Self {
            source_images,
            test_images,

            current_test_image_index: 0,

            v,
            w,
            q,
            t,
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    for i in 0..self.source_images.len() {
                        dbg!(i);
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
