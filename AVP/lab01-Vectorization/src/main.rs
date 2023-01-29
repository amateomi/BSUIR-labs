#![allow(non_snake_case)]

use std::arch::x86_64::*;
use std::time::{Instant};

const LITTLE_SIZE: usize = 16;
const SIZE: usize = 32;

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

fn compare_mega_matrices(m1: &MegaMatrix, m2: &MegaMatrix) -> bool {
    for i in 0..SIZE {
        for j in 0..SIZE {
            for k in 0..LITTLE_SIZE {
                for l in 0..LITTLE_SIZE {
                    if f32::abs(m1[i][j][k][l] - m2[i][j][k][l]) > 0.001 { return false; }
                }
            }
        }
    }

    true
}

fn mul_no_simd(m1: &Matrix, m2: &Matrix) -> Matrix {
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

fn sum_no_simd(result: &mut Matrix, m: Matrix) {
    for i in 0..m.len() {
        for j in 0..m.len() {
            result[i][j] += m[i][j];
        }
    }
}

fn compute_no_simd(result: &mut MegaMatrix, m1: &MegaMatrix, m2: &MegaMatrix) {
    for i in 0..result.len() {
        for j in 0..result.len() {
            for k in 0..result.len() {
                sum_no_simd(&mut result[i][j], mul_no_simd(&m1[i][k], &m2[k][j]));
            }
        }
    }
}

#[target_feature(enable = "sse,avx")]
unsafe fn mul_auto_simd(m1: &Matrix, m2: &Matrix) -> Matrix {
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

#[target_feature(enable = "sse,avx")]
unsafe fn sum_auto_simd(result: &mut Matrix, m: Matrix) {
    for i in 0..m.len() {
        for j in 0..m.len() {
            result[i][j] += m[i][j];
        }
    }
}

fn compute_auto_simd(result: &mut MegaMatrix, m1: &MegaMatrix, m2: &MegaMatrix) {
    for i in 0..result.len() {
        for j in 0..result.len() {
            for k in 0..result.len() {
                unsafe {
                    sum_auto_simd(&mut result[i][j], mul_auto_simd(&m1[i][k], &m2[k][j]));
                }
            }
        }
    }
}

#[target_feature(enable = "sse,avx")]
unsafe fn mul_my_simd(m1: &Matrix, m2: &Matrix) -> Matrix {
    let mut result: Matrix = [[0f32; LITTLE_SIZE]; LITTLE_SIZE];
    for i in 0..m1.len() {
        for j in 0..m1.len() {

            let mat1 = _mm256_set1_ps(m1[i][j]);

            for k in (0..m1.len()).step_by(8) {

                let mat2 = _mm256_loadu_ps(&m2[j][k]);
                let mut multi_res = _mm256_loadu_ps(&result[i][k]);
                multi_res = _mm256_add_ps(multi_res, _mm256_mul_ps(mat1, mat2));

                _mm256_storeu_ps(&mut result[i][k], multi_res);

                // let ymm0 = _mm256_loadu_ps(&m1[i][k]);
                // let ymm1 = _mm256_set_ps(m2[k + 7][j], m2[k + 6][j],
                //                          m2[k + 5][j], m2[k + 4][j],
                //                          m2[k + 3][j], m2[k + 2][j],
                //                          m2[k + 1][j], m2[k + 0][j]);
                // let ymm2 = _mm256_mul_ps(ymm0, ymm1);
                // let high = _mm256_extractf128_ps(ymm2, 1);
                // let sum = _mm_add_ps(_mm256_castps256_ps128(ymm2), high);
                // let sum = _mm_hadd_ps(sum, sum);
                // let sum = _mm_hadd_ps(sum, sum);
                //
                // result[i][j] += _mm_cvtss_f32(sum);
            }
        }
    }
    result
}

#[target_feature(enable = "sse,avx")]
unsafe fn sum_my_simd(result: &mut Matrix, m: Matrix) {
    for i in 0..m.len() {
        for j in (0..m.len()).step_by(8) {
            _mm256_storeu_ps(&mut result[i][j],
                             _mm256_add_ps(_mm256_loadu_ps(&result[i][j]),
                                           _mm256_loadu_ps(&m[i][j])));
        }
    }
}

fn compute_my_simd(result: &mut MegaMatrix, m1: &MegaMatrix, m2: &MegaMatrix) {
    for i in 0..result.len() {
        for j in 0..result.len() {
            for k in 0..result.len() {
                unsafe {
                    sum_my_simd(&mut result[i][j], mul_my_simd(&m1[i][k], &m2[k][j]));
                }
            }
        }
    }
}

fn main() {
    assert_eq!(LITTLE_SIZE % 8, 0);

    let m1 = new_mega_matrix();
    let m2 = new_mega_matrix();

    let mut res1: Box<MegaMatrix> = vec![[[[0f32; LITTLE_SIZE]; LITTLE_SIZE]; SIZE]; SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    let now = Instant::now();
    compute_no_simd(&mut *res1, &*m1, &*m2);
    let elapsed_time = now.elapsed();
    println!("No SIMD took {}ms", elapsed_time.as_millis());

    let mut res2: Box<MegaMatrix> = vec![[[[0f32; LITTLE_SIZE]; LITTLE_SIZE]; SIZE]; SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    let now = Instant::now();
    compute_auto_simd(&mut *res2, &*m1, &*m2);
    let elapsed_time = now.elapsed();
    println!("Auto SIMD took {}ms", elapsed_time.as_millis());

    if compare_mega_matrices(&res1, &res2) {
        println!("res1=res2");
    }

    let mut res3: Box<MegaMatrix> = vec![[[[0f32; LITTLE_SIZE]; LITTLE_SIZE]; SIZE]; SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    let now = Instant::now();
    compute_my_simd(&mut *res3, &*m1, &*m2);
    let elapsed_time = now.elapsed();
    println!("My SIMD took {}ms", elapsed_time.as_millis());

    if compare_mega_matrices(&res1, &res3) {
        println!("res1=res3");
    }
}
