#![allow(non_snake_case)]

use std::arch::x86_64::*;
use std::time::Instant;

const LITTLE_SIZE: usize = 32;
const SIZE: usize = 2;

type Matrix = [[f32; LITTLE_SIZE]; LITTLE_SIZE];
type MegaMatrix = [[Matrix; SIZE]; SIZE];

fn new_mega_matrix() -> Box<MegaMatrix> {
    let mut result: Box<MegaMatrix> = vec![[[[0f32; LITTLE_SIZE]; LITTLE_SIZE]; SIZE]; SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    for row in &mut *result {
        for matrix in row {
            for row in matrix {
                for item in row {
                    *item = rand::random();
                }
            }
        }
    }
    result
}

fn multiply_matrices(m1: &Matrix, m2: &Matrix) -> Matrix {
    let mut result: Matrix = [[0f32; LITTLE_SIZE]; LITTLE_SIZE];
    for i in 0..m1.len() {
        for j in 0..m1.len() {
            for k in 0..m1.len() {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    result
}

fn multiply_matrices_avx(m1: &Matrix, m2: &Matrix) -> Matrix {
    let mut result: Matrix = [[0f32; LITTLE_SIZE]; LITTLE_SIZE];
    let mut buffer = [0f32; 8];
    for i in 0..m1.len() {
        for j in 0..m1.len() {
            unsafe {
                let mut accumulator = _mm256_set1_ps(0f32);
                for k in (0..m1.len()).step_by(8) {
                    let ymm0 = _mm256_loadu_ps(&m1[i][k]);
                    let ymm1 = _mm256_set_ps(m2[k + 7][j], m2[k + 6][j],
                                             m2[k + 5][j], m2[k + 4][j],
                                             m2[k + 3][j], m2[k + 2][j],
                                             m2[k + 1][j], m2[k + 0][j]);
                    accumulator = _mm256_fmadd_ps(ymm0, ymm1, accumulator);
                }
                _mm256_storeu_ps(buffer.as_mut_ptr(), accumulator);
            }
            result[i][j] = buffer.iter().sum();
        }
    }
    result
}

fn sum_matrices(result: &mut Matrix, m: &Matrix) {
    for i in 0..m.len() {
        for j in 0..m.len() {
            result[i][j] += m[i][j];
        }
    }
}

fn compute(result: &mut MegaMatrix, m1: &MegaMatrix, m2: &MegaMatrix) {
    for i in 0..m1.len() {
        for j in 0..m1.len() {
            for k in 0..m1.len() {
                sum_matrices(&mut result[i][j], &multiply_matrices(&m1[i][k], &m2[k][j]));
            }
        }
    }
}

fn compute_avx(result: &mut MegaMatrix, m1: &MegaMatrix, m2: &MegaMatrix) {
    for i in 0..m1.len() {
        for j in 0..m1.len() {
            for k in 0..m1.len() {
                sum_matrices(&mut result[i][j], &multiply_matrices(&m1[i][k], &m2[k][j]));
            }
        }
    }
}

fn compare_matrices(m1: &Matrix, m2: &Matrix) -> bool {
    for i in 0..LITTLE_SIZE {
        for j in 0..LITTLE_SIZE {
            if f32::abs(m1[i][j] - m2[i][j]) > 0.0001 { return false; }
        }
    }
    true
}

fn main() {
    assert_eq!(LITTLE_SIZE % 8, 0);

    let m1: Matrix = rand::random();
    let m2: Matrix = rand::random();

    let res1 = multiply_matrices(&m1, &m2);
    let res2 = multiply_matrices_avx(&m1, &m2);

    if compare_matrices(&res1, &res2) {
        println!("Matrices are the same!");
    }

    // let m1 = new_mega_matrix();
    // let m2 = new_mega_matrix();
    // let mut m3: Box<MegaMatrix> = vec![[[[0f32; LITTLE_SIZE]; LITTLE_SIZE]; SIZE]; SIZE]
    //     .into_boxed_slice()
    //     .try_into()
    //     .unwrap();
    //
    // let now = Instant::now();
    // compute(&mut *m3, &*m1, &*m2);
    // let elapsed_time = now.elapsed();
    // println!("Compute took {}ms", elapsed_time.as_millis());
}
