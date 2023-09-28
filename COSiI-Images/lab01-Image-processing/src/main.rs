use eframe::{ egui, egui::{ plot::BarChart, widgets::plot }, epaint::{ Color32, ColorImage } };
use egui_extras::RetainedImage;
use std::{ fmt, path };

const PICTURE_PATH: &str = "jojo.jpg";

enum ViewState {
    SourceImage,
    Grayscale,
    GrayscaleContrast,
    ColoredContrast,
    LowPassFilterSourceImage,
    LowPassFilterGrayscale,
}

impl fmt::Display for ViewState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewState::SourceImage => write!(f, "source image"),
            ViewState::Grayscale => write!(f, "grayscale"),
            ViewState::GrayscaleContrast => write!(f, "contrast"),
            ViewState::ColoredContrast => write!(f, "colored contrast"),
            ViewState::LowPassFilterSourceImage => write!(f, "low pass filter source image"),
            ViewState::LowPassFilterGrayscale => write!(f, "low pass filter grayscale"),
        }
    }
}

struct ConvolutionKernel {
    coefficient: [usize; 9],
    divider: usize,
}

impl Default for ConvolutionKernel {
    fn default() -> Self {
        Self {
            coefficient: [1; 9],
            divider: 9,
        }
    }
}

trait ImageOperations {
    fn compute_pixel_brightness(pixel: Color32) -> u8;

    fn compute_contrast(
        source_image: &ColorImage,
        source_brightness: &Brightness,
        target_brightness: &Brightness
    ) -> ColorImage {
        let mut result = ColorImage::new(source_image.size, Color32::BLACK);
        for y in 0..source_image.height() {
            for x in 0..source_image.width() {
                let i = y * source_image.width() + x;
                for color_component in 0..3 {
                    let target_color_component = compute_g(
                        source_image.pixels[i][color_component] as u32,
                        source_brightness.min as u32,
                        source_brightness.max as u32,
                        target_brightness.min as u32,
                        target_brightness.max as u32
                    );
                    result.pixels[i][color_component] = target_color_component;
                }
            }
        }
        result
    }

    fn compute_brightness(image: &ColorImage) -> Brightness {
        let mut result = Brightness::default();
        for y in 0..image.height() {
            for x in 0..image.width() {
                let i = y * image.width() + x;

                let brightness = Self::compute_pixel_brightness(image.pixels[i]);
                if result.max < brightness {
                    result.max = brightness;
                } else if result.min > brightness {
                    result.min = brightness;
                }
            }
        }
        result
    }

    fn find_bar_chart(image: &ColorImage) -> Vec<plot::Bar> {
        let mut bar_chart: [usize; (u8::MAX as usize) + 1] = [0; (u8::MAX as usize) + 1];
        for y in 0..image.height() {
            for x in 0..image.width() {
                let i = y * image.width() + x;

                bar_chart[Self::compute_pixel_brightness(image.pixels[i]) as usize] += 1;
            }
        }
        bar_chart
            .iter()
            .enumerate()
            .map(|(x, y)| plot::Bar::new(x as f64, *y as f64))
            .collect()
    }

    fn low_pass_filter_for_pixel_component(
        image: &ColorImage,
        target_x: usize,
        target_y: usize,
        kernel: &ConvolutionKernel,
        color_component_index: usize
    ) -> u8 {
        let mut accumulator: usize = 0;
        for y in 0..3 {
            for x in 0..3 {
                let i = (y + target_y - 1) * image.width() + (x + target_x - 1);
                let kernel_index = y * 3 + x;
                accumulator +=
                    (image.pixels[i][color_component_index] as usize) *
                    kernel.coefficient[kernel_index];
            }
        }
        let result = accumulator / kernel.divider;
        if result > (u8::MAX as usize) {
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
                for color_component in 0..3 {
                    let target_color_component = Self::low_pass_filter_for_pixel_component(
                        image,
                        x,
                        y,
                        &kernel,
                        color_component
                    );
                    result.pixels[i][color_component] = target_color_component;
                }
            }
        }
        result
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Brightness {
    min: u8,
    max: u8,
}

impl Default for Brightness {
    fn default() -> Self {
        Self { min: 0, max: 255 }
    }
}

struct ColoredImage {
    data: ColorImage,
    brightness: Brightness,
    bar_chart: Vec<plot::Bar>,
}

impl ImageOperations for ColoredImage {
    fn compute_pixel_brightness(pixel: Color32) -> u8 {
        let red = pixel.r() as f32;
        let green = pixel.g() as f32;
        let blue = pixel.b() as f32;

        let result = 0.3 * red + 0.59 * green + 0.11 * blue;

        if result > (u8::MAX as f32) {
            u8::MAX
        } else {
            result.round() as u8
        }
    }
}

impl ColoredImage {
    fn new(data: ColorImage) -> Self {
        let brightness = Self::compute_brightness(&data);
        let bar_chart = Self::find_bar_chart(&data);
        Self { data, brightness, bar_chart }
    }
}

struct GrayscaleImage {
    data: ColorImage,
    brightness: Brightness,
    bar_chart: Vec<plot::Bar>,
}

impl ImageOperations for GrayscaleImage {
    fn compute_pixel_brightness(pixel: Color32) -> u8 {
        let red = pixel.r();
        let green = pixel.g();
        let blue = pixel.b();
        if red != green && green != blue {
            panic!("pixel not gray: RGB[{red},{green},{blue}]");
        }
        red
    }
}

impl GrayscaleImage {
    fn new(source_image: &ColorImage) -> Self {
        let mut data = ColorImage::new(source_image.size, Color32::BLACK);
        for y in 0..source_image.height() {
            for x in 0..source_image.width() {
                let i = y * source_image.width() + x;

                let gray_level = ColoredImage::compute_pixel_brightness(source_image.pixels[i]);

                data.pixels[i] = Color32::from_gray(gray_level);
            }
        }
        let brightness = Self::compute_brightness(&data);
        let bar_chart = Self::find_bar_chart(&data);
        Self { data, brightness, bar_chart }
    }
}

type BrightnessInputField = String;

struct Application {
    state: ViewState,

    source_image: ColoredImage,
    grayscale_image: GrayscaleImage,

    contrast_image: Option<ColoredImage>,
    grayscale_contrast_image: Option<GrayscaleImage>,

    antialiased_image: ColoredImage,
    antialiased_grayscale_image: GrayscaleImage,

    target_brightness: Brightness,
    min_target_brightness_input: BrightnessInputField,
    max_target_brightness_input: BrightnessInputField,
}

impl Default for Application {
    fn default() -> Self {
        let source_image = ColoredImage::new(
            load_image_from_path(path::Path::new(PICTURE_PATH)).unwrap()
        );
        let grayscale_image = GrayscaleImage::new(&source_image.data);

        let antialiased_image = ColoredImage::new(
            ColoredImage::compute_low_pass_filter(&source_image.data)
        );
        let antialiased_grayscale_image = GrayscaleImage::new(
            &GrayscaleImage::compute_low_pass_filter(&grayscale_image.data)
        );

        Self {
            state: ViewState::SourceImage,

            source_image,
            grayscale_image,

            contrast_image: None,
            grayscale_contrast_image: None,

            antialiased_image,
            antialiased_grayscale_image,

            target_brightness: Brightness::default(),
            min_target_brightness_input: BrightnessInputField::new(),
            max_target_brightness_input: BrightnessInputField::new(),
        }
    }
}

impl eframe::App for Application {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered_justified(|ui| {
                if ui.button("<-").clicked() {
                    self.state = match self.state {
                        ViewState::SourceImage => ViewState::LowPassFilterGrayscale,
                        ViewState::Grayscale => ViewState::SourceImage,
                        ViewState::GrayscaleContrast => ViewState::Grayscale,
                        ViewState::ColoredContrast => ViewState::GrayscaleContrast,
                        ViewState::LowPassFilterSourceImage => ViewState::ColoredContrast,
                        ViewState::LowPassFilterGrayscale => ViewState::LowPassFilterSourceImage,
                    };
                }
                if ui.button("->").clicked() {
                    self.state = match self.state {
                        ViewState::SourceImage => ViewState::Grayscale,
                        ViewState::Grayscale => ViewState::GrayscaleContrast,
                        ViewState::GrayscaleContrast => ViewState::ColoredContrast,
                        ViewState::ColoredContrast => ViewState::LowPassFilterSourceImage,
                        ViewState::LowPassFilterSourceImage => ViewState::LowPassFilterGrayscale,
                        ViewState::LowPassFilterGrayscale => ViewState::SourceImage,
                    };
                }
            });

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading(format!("{}", self.state));
                    match self.state {
                        ViewState::SourceImage => {
                            ui.label(format!("{:?}", self.source_image.brightness));
                        }
                        ViewState::Grayscale => {
                            ui.label(format!("{:?}", self.grayscale_image.brightness));
                        }
                        ViewState::GrayscaleContrast | ViewState::ColoredContrast => {
                            ui.horizontal(|ui| {
                                ui.label("Brightness:");

                                let mut input_brightness = self.target_brightness.clone();
                                ui.label("min: ");
                                if
                                    ui
                                        .text_edit_singleline(&mut self.min_target_brightness_input)
                                        .changed()
                                {
                                    if
                                        let Ok(min_brightness) =
                                            self.min_target_brightness_input.parse::<u8>()
                                    {
                                        input_brightness.min = min_brightness;
                                    }
                                }
                                ui.label("max: ");
                                if
                                    ui
                                        .text_edit_singleline(&mut self.max_target_brightness_input)
                                        .changed()
                                {
                                    if
                                        let Ok(max_brightness) =
                                            self.max_target_brightness_input.parse::<u8>()
                                    {
                                        input_brightness.max = max_brightness;
                                    }
                                }

                                if
                                    input_brightness.min < input_brightness.max &&
                                    input_brightness != self.target_brightness
                                {
                                    self.target_brightness = input_brightness;
                                    match self.state {
                                        ViewState::GrayscaleContrast => {
                                            self.grayscale_contrast_image = Some(
                                                GrayscaleImage::new(
                                                    &GrayscaleImage::compute_contrast(
                                                        &self.grayscale_image.data,
                                                        &self.grayscale_image.brightness,
                                                        &self.target_brightness
                                                    )
                                                )
                                            );
                                        }
                                        ViewState::ColoredContrast => {
                                            self.contrast_image = Some(
                                                ColoredImage::new(
                                                    ColoredImage::compute_contrast(
                                                        &self.source_image.data,
                                                        &self.source_image.brightness,
                                                        &self.target_brightness
                                                    )
                                                )
                                            );
                                        }
                                        _ => { panic!("Impossible logic state") }
                                    }
                                }
                            });
                        }
                        _ => {}
                    }
                });

                ui.horizontal(|ui| {
                    let (name, image, bar_chart) = match self.state {
                        ViewState::SourceImage =>
                            (
                                "source image",
                                self.source_image.data.clone(),
                                self.source_image.bar_chart.clone(),
                            ),
                        ViewState::Grayscale =>
                            (
                                "grayscale image",
                                self.grayscale_image.data.clone(),
                                self.grayscale_image.bar_chart.clone(),
                            ),
                        ViewState::GrayscaleContrast => {
                            if let Some(image) = &self.grayscale_contrast_image {
                                ("contrast image", image.data.clone(), image.bar_chart.clone())
                            } else {
                                (
                                    "default grayscale image",
                                    self.grayscale_image.data.clone(),
                                    self.grayscale_image.bar_chart.clone(),
                                )
                            }
                        }
                        ViewState::ColoredContrast => {
                            if let Some(image) = &self.contrast_image {
                                (
                                    "colored contrast image",
                                    image.data.clone(),
                                    image.bar_chart.clone(),
                                )
                            } else {
                                (
                                    "default colored image",
                                    self.source_image.data.clone(),
                                    self.source_image.bar_chart.clone(),
                                )
                            }
                        }
                        ViewState::LowPassFilterGrayscale =>
                            (
                                "low pass filter grayscale image",
                                self.antialiased_grayscale_image.data.clone(),
                                self.antialiased_grayscale_image.bar_chart.clone(),
                            ),
                        ViewState::LowPassFilterSourceImage =>
                            (
                                "low pass filter colored image",
                                self.antialiased_image.data.clone(),
                                self.antialiased_image.bar_chart.clone(),
                            ),
                    };
                    RetainedImage::from_color_image(name, image).show(ui);
                    plot::Plot::new(name).show(ui, |plot_ui| {
                        plot_ui.bar_chart(BarChart::new(bar_chart));
                    });
                });
            });
        });
    }
}

fn load_image_from_path(path: &path::Path) -> Result<egui::ColorImage, image::ImageError> {
    let image = image::io::Reader::open(path)?.decode()?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgba8();
    let pixels = image_buffer.as_flat_samples();
    Ok(egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()))
}

fn compute_g(f_cur: u32, f_min: u32, f_max: u32, g_min: u32, g_max: u32) -> u8 {
    let result = ((f_cur - f_min) * (g_max - g_min)) / (f_max - f_min) + g_min;
    if result > (u8::MAX as u32) {
        u8::MAX
    } else {
        result as u8
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        initial_window_size: Option::from(egui::Vec2::new(1000f32, 700f32)),
        ..Default::default()
    };
    eframe::run_native(
        "Image processing",
        options,
        Box::new(|_cc| Box::<Application>::default())
    )
}
