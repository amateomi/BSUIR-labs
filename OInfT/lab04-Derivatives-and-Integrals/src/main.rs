use core::array::from_fn;

const A: usize = 5;
const B: usize = 8;
const TOTAL_POINTS: usize = 21;

fn f(x: f64) -> f64 {
    x.sqrt() - x.cos().powi(2)
}

fn f_d1(x: f64) -> f64 {
    1.0 / (2.0 * x.sqrt()) + (2.0 * x).sin()
}

fn f_d2(x: f64) -> f64 {
    -1.0 / (4.0 * x.powf(3.0 / 2.0)) + 2.0 * (2.0 * x).cos()
}

fn f_d1_approx(x: f64, h: f64) -> f64 {
    let y1 = f(x - h);
    let y3 = f(x + h);
    (y3 - y1) / (2.0 * h)
}

fn f_d2_approx(x: f64, h: f64) -> f64 {
    let y1 = f(x - h);
    let y2 = f(x);
    let y3 = f(x + h);
    (y1 - 2.0 * y2 + y3) / h.powi(2)
}

fn f_i(m: i32) -> f64 {
    let h = (B - A) as f64 / m as f64;
    let mut s = 0.0;
    let mut x = (A as f64) + h / 2.0;
    for _ in 0..m {
        let node_1 = x - h * 0.5773502692;
        let node_2 = x + h * 0.5773502692;
        s += h / 2.0 * (f(node_1) + f(node_2));
        x += h;
    }
    s
}

fn main() {
    let x: [f64; TOTAL_POINTS] = from_fn(|i| A as f64 + i as f64 * (B - A) as f64 / 20.0);
    println!("x: {x:?}");
    let f: [f64; TOTAL_POINTS] = from_fn(|i| f(x[i]));
    println!("f: {f:?}");

    let f_d1: [f64; TOTAL_POINTS] = from_fn(|i| f_d1(x[i]));
    println!("f': {f_d1:?}");
    let f_d2: [f64; TOTAL_POINTS] = from_fn(|i| f_d2(x[i]));
    println!("f'': {f_d2:?}");

    for h in [0.2, 0.1, 0.05] {
        let f_d1_approx: [f64; TOTAL_POINTS] = from_fn(|i| f_d1_approx(x[i], h));
        println!("h={h}, approximation of f': {f_d1_approx:?}");
        let f_d2_approx: [f64; TOTAL_POINTS] = from_fn(|i| f_d2_approx(x[i], h));
        println!("h={h}, approximation of f'': {f_d2_approx:?}");
    }

    println!("integral of f on [{A}, {B}]: 6.067");
    for m in [10, 20, 40] {
        let s = f_i(m);
        println!("m={m}, approximation of integral of f on [{A}, {B}]: {s}");
    }
}
