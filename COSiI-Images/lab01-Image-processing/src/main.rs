use eframe::{
    egui,
    egui::{widgets::plot, plot::BarChart},
    epaint::{Color32, ColorImage},
};
use egui_extras::RetainedImage;
use std::{fmt, path};

#[derive(PartialEq)]
enum ViewState {
    SourceImage,
    GrayscaleImage,
    ContrastImage,
    ColoredContrastImage,
    LowPassFilterSourceImage,
    LowPassFilterGrayscaleImage,
}

impl fmt::Display for ViewState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewState::SourceImage => write!(f, "source image"),
            ViewState::GrayscaleImage => write!(f, "grayscale image"),
            ViewState::ContrastImage => write!(f, "contrast image"),
            ViewState::ColoredContrastImage => write!(f, "colored contrast image"),
            ViewState::LowPassFilterSourceImage => write!(f, "low pass filter source image"),
            ViewState::LowPassFilterGrayscaleImage => write!(f, "low pass filter grayscale image"),
        }
    }
}

#[derive(Clone, Debug)]
struct ConvolutionKernel {
    coefficient: [usize; 9],
    divider: usize,
}

impl Default for ConvolutionKernel {
    fn default() -> Self {
        Self { coefficient: [1; 9], divider: 9}
    }
}

#[derive(Debug)]
struct Brightness {
    min: u8,
    max: u8,
}

impl Default for Brightness {
    fn default() -> Self {
        Self { min: 0, max: 255 }
    }
}

struct Application {
    state: ViewState,
    
    source_image: ColorImage,
    gray_image: ColorImage,

    source_brightness_bar_chart: Vec<plot::Bar>,

    source_brightness: Brightness,
    target_brightness: Brightness,

    min_target_brightness_buffer: String,
    max_target_brightness_buffer: String,
}

impl Default for Application {
    fn default() -> Self {
        let source_image = load_image_from_path(path::Path::new("jojo.jpg")).unwrap();
        let gray_image = image_to_gray(&source_image);

        let source_brightness = find_brightness(&source_image);

        let source_brightness_bar_chart = find_bar_chart(&source_image);

        Self {
            state: ViewState::SourceImage,
                
            source_image,
            gray_image,

            source_brightness_bar_chart: source_brightness_bar_chart.iter().enumerate().map(|(x, y)| plot::Bar::new(x as f64, *y as f64)).collect(),

            source_brightness,
            target_brightness: Brightness::default(),
            
            min_target_brightness_buffer: String::new(),
            max_target_brightness_buffer: String::new(),
        }
    }
}

fn find_brightness(image: &ColorImage) -> Brightness {
    let mut result = Brightness::default();

    for y in 0..image.height() {
        for x in 0..image.width() {
            let i = y * image.width() + x;

            let px_brightness = compute_pixel_brightness(image.pixels[i]);
            if result.max < px_brightness {
                result.max = px_brightness;
            } else if result.min > px_brightness {
                result.min = px_brightness;
            }
        }
    }

    result
}

fn find_bar_chart(image: &ColorImage) -> [usize; u8::MAX as usize + 1] {
    let mut bar_chart: [usize; u8::MAX as usize + 1] = [0; u8::MAX as usize + 1];

    for y in 0..image.height() {
        for x in 0..image.width() {
            let i = y * image.width() + x;

            bar_chart[compute_pixel_brightness(image.pixels[i]) as usize] += 1;
        }
    }

    bar_chart
}

fn compute_g(f_cur: u32, f_min: u32, f_max: u32, g_min: u32, g_max: u32) -> u8 {
    let result = ((f_cur - f_min) * (g_max - g_min)) / (f_max - f_min) + g_min;
    if result > u8::MAX as u32 {
        u8::MAX
    } else {
        result as u8
    }
}

fn compute_colored_contrast_image(
    image: &ColorImage,
    brightness: &Brightness,
    target_brightness: &Brightness,
) -> ColorImage {
    let mut result = ColorImage::new(image.size, Color32::BLACK);

    for y in 0..image.height() {
        for x in 0..image.width() {
            let i = y * image.width() + x;

            let red = image.pixels[i].r();
            let green = image.pixels[i].g();
            let blue = image.pixels[i].b();

            let target_red = compute_g(
                red as u32,
                brightness.min as u32,
                brightness.max as u32,
                target_brightness.min as u32,
                target_brightness.max as u32,
            );
            let target_green = compute_g(
                green as u32,
                brightness.min as u32,
                brightness.max as u32,
                target_brightness.min as u32,
                target_brightness.max as u32,
            );
            let target_blue = compute_g(
                blue as u32,
                brightness.min as u32,
                brightness.max as u32,
                target_brightness.min as u32,
                target_brightness.max as u32,
            );

            result.pixels[i] = Color32::from_rgb(target_red, target_green, target_blue);
        }
    }

    result
}

fn compute_contrast_image(
    image: &ColorImage,
    brightness: &Brightness,
    target_brightness: &Brightness,
) -> ColorImage {
    let mut result = ColorImage::new(image.size, Color32::BLACK);

    for y in 0..image.height() {
        for x in 0..image.width() {
            let i = y * image.width() + x;

            let gray_level = image.pixels[i].r();

            let target_gray_level = compute_g(
                gray_level as u32,
                brightness.min as u32,
                brightness.max as u32,
                target_brightness.min as u32,
                target_brightness.max as u32,
            );

            result.pixels[i] = Color32::from_gray(target_gray_level);
        }
    }

    result
}

fn low_pass_filter_for_pixel_component(image: &ColorImage, target_x: usize, target_y: usize, kernel: &ConvolutionKernel, color_component_index: usize) -> u8 {
    let mut accumulator: usize = 0;
    for y in 0..3usize {
        for x in 0..3usize {
            let i = (y + target_y - 1) * image.width() + (x + target_x - 1);
            let kernel_index = y * 3 + x;
            accumulator += image.pixels[i][color_component_index] as usize * kernel.coefficient[kernel_index];
        }
    }
    let result = accumulator / kernel.divider;
    if result > u8::MAX as usize {
        u8::MAX
    } else {
        result as u8
    }
}

fn compute_low_pass_filter(image: &ColorImage) -> ColorImage {

    let mut result = ColorImage::new(image.size, Color32::BLACK);

    let kernel = ConvolutionKernel::default();

    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            let i = y * image.width() + x;

            let red = low_pass_filter_for_pixel_component(image, x, y, &kernel, 0);
            let green = low_pass_filter_for_pixel_component(image, x, y, &kernel, 1);
            let blue = low_pass_filter_for_pixel_component(image, x, y, &kernel, 2);

            result.pixels[i] = Color32::from_rgb(red, green, blue);
        }
    }

    result
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("<-").clicked() {
                    self.state = match self.state {
                        ViewState::SourceImage => ViewState::LowPassFilterGrayscaleImage,
                        ViewState::GrayscaleImage => ViewState::SourceImage,
                        ViewState::ContrastImage => ViewState::GrayscaleImage,
                        ViewState::ColoredContrastImage => ViewState::ContrastImage,
                        ViewState::LowPassFilterSourceImage => ViewState::ColoredContrastImage,
                        ViewState::LowPassFilterGrayscaleImage => ViewState::LowPassFilterSourceImage,
                    };
                }
                if ui.button("->").clicked() {
                    self.state = match self.state {
                        ViewState::SourceImage => ViewState::GrayscaleImage,
                        ViewState::GrayscaleImage => ViewState::ContrastImage,
                        ViewState::ContrastImage => ViewState::ColoredContrastImage,
                        ViewState::ColoredContrastImage => ViewState::LowPassFilterSourceImage,
                        ViewState::LowPassFilterSourceImage => ViewState::LowPassFilterGrayscaleImage,
                        ViewState::LowPassFilterGrayscaleImage => ViewState::SourceImage,
                    };
                }
            });

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading(format!("{0}", self.state));
                    
                    match self.state {
                        ViewState::SourceImage | ViewState::GrayscaleImage => {
                            ui.label(format!("{:?}", self.source_brightness));
                        },
                        ViewState::ContrastImage | ViewState::ColoredContrastImage => {
                            ui.horizontal(|ui| {
                                ui.label("Brightness:");

                                ui.label("min: ");
                                if ui.text_edit_singleline(&mut self.min_target_brightness_buffer).changed() {
                                    if let Ok(min_brightness) =
                                        self.min_target_brightness_buffer.parse::<u8>()
                                    {
                                        self.target_brightness.min = min_brightness;
                                    }
                                }
                                
                                ui.label("max: ");
                                if ui.text_edit_singleline(&mut self.max_target_brightness_buffer).changed() {
                                    if let Ok(max_brightness) = self.max_target_brightness_buffer.parse::<u8>() {
                                        self.target_brightness.max = max_brightness;
                                    }
                                }
                            });
                        }
                        ViewState::LowPassFilterSourceImage | ViewState::LowPassFilterGrayscaleImage => ()
                    }
                });

                ui.horizontal(|ui| {
                    match self.state {
                        ViewState::SourceImage => {
                            RetainedImage::from_color_image("source image", self.source_image.clone())
                                .show(ui)
                        }
                        ViewState::GrayscaleImage => {
                            RetainedImage::from_color_image("gray image", self.gray_image.clone())
                                .show(ui)
                        }
                        ViewState::ContrastImage | ViewState::ColoredContrastImage => {
                            if self.target_brightness.min >= self.target_brightness.max {
                                ui.heading(format!(
                                    "Target brightness minimal value={} must be lower than maximum value={}",
                                    self.target_brightness.min, self.target_brightness.max
                                ))
                            } else {
                                let image = match self.state {
                                    ViewState::ContrastImage => compute_contrast_image(
                                        &self.gray_image,
                                        &self.source_brightness,
                                        &self.target_brightness,
                                    ),
                                    ViewState::ColoredContrastImage => compute_colored_contrast_image(
                                        &self.source_image,
                                         &self.source_brightness,
                                          &self.target_brightness
                                    ),
                                    _ => panic!("Unexpected state"),
                                };
                                RetainedImage::from_color_image("contrast image", image).show(ui)
                            }
                        }
                        ViewState::LowPassFilterSourceImage => {
                            let image = compute_low_pass_filter(&self.source_image);
                            RetainedImage::from_color_image("low pass filter source image", image).show(ui)
                        }
                        ViewState::LowPassFilterGrayscaleImage => {
                            let image = compute_low_pass_filter(&self.gray_image);
                            RetainedImage::from_color_image("low pass filter target image", image).show(ui)
                        }
                    };
                    match self.state {
                        ViewState::SourceImage | ViewState::GrayscaleImage => {
                            let plot = plot::Plot::new("source image brightness bar chart");
                            plot.show(ui, |plot_ui| {
                                plot_ui.bar_chart(BarChart::new(self.source_brightness_bar_chart.clone()))
                            });
                        }
                        ViewState::ContrastImage => {
                            if self.target_brightness.min >= self.target_brightness.max {
                                ui.heading(format!(
                                    "Target brightness minimal value={} must be lower than maximum value={}",
                                    self.target_brightness.min, self.target_brightness.max
                                ));
                            } else {
                            let plot = plot::Plot::new("target image brightness bar chart");
                            let chart = find_bar_chart(&compute_contrast_image(
                                &self.gray_image,
                                &self.source_brightness,
                                &self.target_brightness,
                            )).iter().enumerate().map(|(x, y)| plot::Bar::new(x as f64, *y as f64)).collect();
                            plot.show(ui, |plot_ui| {
                                plot_ui.bar_chart(BarChart::new(chart))
                            });
                        }
                        }
                        ViewState::ColoredContrastImage => {
                            if self.target_brightness.min >= self.target_brightness.max {
                                ui.heading(format!(
                                    "Target brightness minimal value={} must be lower than maximum value={}",
                                    self.target_brightness.min, self.target_brightness.max
                                ));
                            } else {
                            let plot = plot::Plot::new("target image brightness bar chart");
                            let chart = find_bar_chart(&compute_colored_contrast_image(
                                &self.source_image,
                                &self.source_brightness,
                                &self.target_brightness,
                            )).iter().enumerate().map(|(x, y)| plot::Bar::new(x as f64, *y as f64)).collect();
                            plot.show(ui, |plot_ui| {
                                plot_ui.bar_chart(BarChart::new(chart))
                            });
                        }
                        }
                        _ => ()
                    }
                })
            });
        });
    }
}

fn load_image_from_path(path: &path::Path) -> Result<egui::ColorImage, image::ImageError> {
    let image = image::io::Reader::open(path)?.decode()?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgba8();
    let pixels = image_buffer.as_flat_samples();
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        pixels.as_slice(),
    ))
}

fn compute_pixel_brightness(pixel: Color32) -> u8 {
    let red = pixel.r() as f32;
    let green = pixel.g() as f32;
    let blue = pixel.b() as f32;

    let result = 0.3 * red + 0.59 * green + 0.11 * blue;
    if result > u8::MAX as f32 {
        u8::MAX
    } else {
    result.round() as u8
    }
}

fn image_to_gray(image: &ColorImage) -> ColorImage {
    let mut result = ColorImage::new(image.size, Color32::BLACK);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let i = y * image.width() + x;

            let gray_level = compute_pixel_brightness(image.pixels[i]);

            result.pixels[i] = Color32::from_gray(gray_level);
        }
    }
    result
}

fn main() -> eframe::Result<()> {
    let mut options = eframe::NativeOptions::default();
    options.initial_window_size = Option::from(egui::Vec2::new(1000f32, 700f32));
    eframe::run_native(
        "Image processing",
        options,
        Box::new(|_cc| Box::<Application>::default()),
    )
}
