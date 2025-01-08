use plotters::prelude::*;

const A: usize = 5;
const B: usize = 8;
const M: usize = 4;
const N: usize = 4;

const TOTAL_POINTS: usize = 21;

fn draw_diagram(table: &[(f64, f64)], file_name: &str, title: &str) {
    let root = BitMapBackend::new(file_name, (500, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 36).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d((A as f64)..(B as f64), 1.0..3.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(table.to_vec(), &RED))
        .unwrap();

    root.present().unwrap();
}

fn pol(table: &[(f64, f64)]) -> [f64; N] {
    let mut a = <[[f64; N]; N]>::default();
    let mut c = <[f64; N]>::default();
    for k in 1..=N {
        a[k - 1][0] = 1.0;
        let mut i = 0;
        for m in 2..=N {
            i += 1;
            if i == k {
                i += 1;
            }
            let d = table[k - 1].0 - table[i - 1].0;
            a[k - 1][m - 1] = a[k - 1][m - 1 - 1] / d;
            let mut j = m - 1;
            while j >= 2 {
                a[k - 1][j - 1] = (a[k - 1][j - 1 - 1] - a[k - 1][j - 1] * table[i - 1].0) / d;
                j -= 1;
            }
            a[k - 1][0] = -a[k - 1][0] * table[i - 1].0 / d;
        }
    }
    for i in 1..=N {
        c[i - 1] = 0.0;
        for k in 1..=N {
            c[i - 1] += a[k - 1][i - 1] * table[k - 1].1;
        }
    }
    c
}

fn main() {
    let function = |x: f64| x.sqrt() - x.cos().powi(2);
    let table_filler = |i, m: f64| {
        let i = (i + 1) as f64;
        let a = A as f64;
        let b = B as f64;
        let x = a + (i - 1.0) * (b - a) / (m - 1.0);
        let y = function(x);
        (x, y)
    };
    let table_base: [(f64, f64); M] = core::array::from_fn(|i| table_filler(i, M as f64));
    println!("table_base={table_base:?}");
    draw_diagram(
        table_base.as_slice(),
        "table_base.png",
        "График функции f(x) по M точкам",
    );

    let table_reference: [(f64, f64); TOTAL_POINTS] =
        core::array::from_fn(|i| table_filler(i, TOTAL_POINTS as f64));
    println!("table_reference={table_reference:?}");
    draw_diagram(
        table_reference.as_slice(),
        "table_reference.png",
        "График функции f(x) по 21 точке",
    );

    let c = pol(table_base.as_slice());
    println!("c={c:?}");

    let function_approx = |x: f64| (0..c.len()).fold(0.0, |acc, i| acc + c[i] * x.powi(i as i32));
    let table_approx: [(f64, f64); TOTAL_POINTS] = core::array::from_fn(|i| {
        let x = table_reference[i].0;
        let y = function_approx(x);
        (x, y)
    });
    println!("table_approx={table_approx:?}");
    draw_diagram(
        table_approx.as_slice(),
        "table_approx.png",
        "График аппроксимирующей функции",
    );

    let d: [f64; TOTAL_POINTS] = core::array::from_fn(|i| table_reference[i].1 - table_approx[i].1);
    println!("d={d:?}");
}
