const N: usize = 4;
const N1: usize = N + 1;

fn main() {
    let a = [
        [-4.67, 1.0, 0.0, 0.0, 0.0],
        [1.0, -2.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, -2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, -2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, -4.67],
    ];
    let mut x = [0.0; N1];
    let b = [0.0, -3.0, -3.0, -3.0, 0.0];

    let q = [a[0][0], a[1][1], a[2][2], a[3][3], a[4][4]];
    let p = [a[1][0], a[2][1], a[3][2], a[4][3]];
    let r = [a[0][1], a[1][2], a[2][3], a[3][4]];
    let d = b;
    let mut e = [0.0; N];
    let mut n = [0.0; N];

    e[0] = -r[0] / q[0];
    n[0] = d[0] / q[0];
    for i in 2..N1 {
        let z = q[i - 1] + p[i - 1] * e[i - 2];
        e[i - 1] = -r[i - 1] / z;
        n[i - 1] = (d[i - 1] - p[i - 1] * n[i - 2]) / z;
    }
    x[N] = (d[N] - p[N - 1] * n[N - 1]) / (q[N] + p[N - 1] * e[N - 1]);
    for i in (1..N1).rev() {
        x[i - 1] = e[i - 1] * x[i] + n[i - 1];
    }
    println!("x={x:?}");

    let mut discrepancy = 0.0;
    for i in 0..N1 {
        let mut sum = 0.0;
        for j in 0..N1 {
            sum += a[i][j] * x[j];
        }
        let temp = f64::abs(b[i] - sum);
        discrepancy = f64::max(discrepancy, temp);
    }
    println!("∆={discrepancy}");
}
