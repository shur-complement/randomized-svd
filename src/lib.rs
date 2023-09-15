//! Randomized Singular Value Decomposition
//!
//! Based on "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions"
//! Halko et. al
//! https://arxiv.org/abs/0909.4061
//!

use nalgebra::{DMatrix, DVector, SVD};
use nalgebra_sparse::csr::CsrMatrix;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Ranomized Singular Value Decomposition of a matrix.
#[derive(Clone)]
pub struct RSVD {
    /// The left-singular vectors `U` of this SVD.
    pub u: DMatrix<f32>,
    /// The right-singular vectors `V^t` of this SVD.
    pub v_t: DMatrix<f32>,
    /// The singular values of this SVD.
    pub singular_values: DVector<f32>,
}

/// Randomized SVD
/// 
/// As seen in (p. 227 of Halko et al).
///
/// * `A` - (m x n) matrix.
/// * `rank` - Desired rank approximation.
/// * `n_oversamples` - Oversampling parameter for Gaussian random samples.
/// * `n_subspace_iters` - Number of power iterations.
/// * return  U, S as in truncated SVD.
/// 
/// Users should prefer to choose a larger number of samples, as subspace iterations are costly.
/// the default is 2*rank
/// 
pub fn rsvd(
    A: &CsrMatrix<f32>,
    rank: usize,
    n_oversamples: Option<usize>,
    n_subspace_iters: Option<usize>,
    mut rng: &mut impl Rng,
) -> RSVD {

    let n_samples = match n_oversamples {
        None => 2 * rank,
        Some(n) => rank + n,
    };

    // stage A
    let Q = find_range(A, n_samples, n_subspace_iters, &mut rng);

    // stage B
    let B = mul_AT_by_B(&Q, A);
    let SVD {
        u, singular_values, v_t,
    } = B.svd(true, true);
    let ũ = u.unwrap();
    let U_tmp = Q * ũ;

    let v_t = v_t.unwrap();
    
    // truncate
    let U = U_tmp.index((.., ..rank));
    let S = singular_values.index((..rank, ..));
    let Vt = v_t.index((..rank, ..));

    RSVD {
        u: U.into(),
        singular_values: S.into(),
        v_t: Vt.into(),
    }

}

/// Given a matrix A and a number of samples,
/// computes an orthonormal matrix that approximates the range of A.
fn find_range(
    A: &CsrMatrix<f32>,
    n_samples: usize,
    n_subspace_iters: Option<usize>,
    mut rng: &mut impl Rng,
) -> DMatrix<f32> {

    let N = A.ncols();
    let O = DMatrix::from_fn(N, n_samples, |_, _| StandardNormal.sample(&mut rng));
    let Y = A * O;
    match n_subspace_iters {
        Some(iters) => subspace_iter(&A, Y, iters),
        None => ortho_basis(Y),
    }
}

/// Computes orthonormal basis of matrix M
fn ortho_basis(M: DMatrix<f32>) -> DMatrix<f32> {
    let qr = M.qr();
    qr.q()
}

/// Computes orthonormalized approximate range of A
/// after power iterations.
fn subspace_iter(A: &CsrMatrix<f32>, Y0: DMatrix<f32>, n_iters: usize) -> DMatrix<f32> {
    let mut Q = ortho_basis(Y0);
    for _ in 0..n_iters {
        let Z = ortho_basis(A.transpose() * &Q);
        Q = ortho_basis(A * Z);
    }
    return Q;
}

/// dense-sparse product
/// multiplies a dense row-major matrix by a compresed sparse row matrix
/// Writes output in Column-Major order
fn dnsrow_csr_matmul(
    a_nrows: usize,
    b_nrows: usize,
    a_data: &[f32],
    b_data: &[f32],
    b_indptr: &[usize],
    b_indices: &[usize],
    c: &mut [f32],
) {
    for i in 0..a_nrows {
        for j in 0..b_nrows {
            for k in b_indptr[j]..b_indptr[j + 1] {
                let l = b_indices[k];
                c[l * a_nrows + i] += a_data[i * b_nrows + j] * b_data[k];
            }
        }
    }
}

/// Compute A.T * B
/// Allocates a new matrix in column-major order 
fn mul_AT_by_B(A: &DMatrix<f32>, B: &CsrMatrix<f32>) -> DMatrix<f32> {
    // Since A is in column-major form
    // simply reinterpret the data to be
    // in row-major form by swapping axes
    let a_nrows = A.ncols();
    let b_nrows = B.nrows();
    let b_ncols = B.ncols();
    let (b_indptr, b_indices, b_data) = B.csr_data();
    let mut c_data = vec![0.0; a_nrows*b_ncols];
    dnsrow_csr_matmul(a_nrows, b_nrows, A.as_slice(), b_data, b_indptr, b_indices, &mut c_data);
    DMatrix::from_vec(a_nrows, b_ncols, c_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;

    fn get_csr() -> CsrMatrix<f32> {
        let indptr = vec![ 0 ,2, 3, 6, 7, 8, 9, 10];
        let indices = vec![1,2,0,0,3,4,2,2,6,5];
        let data = vec![0.5, 1.,1.,0.33333334, 0.66666669, 1., 1., 1.,1.,1.,];
        let A = CsrMatrix::try_from_csr_data(
            7,7,
            indptr,
            indices,
            data,
        ).unwrap();
        return A;
    }

    #[test]
    fn test_rsvd() {
        let A = get_csr();
        let mut rng = StdRng::seed_from_u64(1337);
        let RSVD{u, singular_values, v_t} = rsvd(&A, 3, Some(3), Some(1), &mut rng);
        let u_expected = DMatrix::from_column_slice(
            7, 3,
            &[-0.60979235, -5.0159326e-8, 3.8545048e-7,
              -0.5604257, -0.56042564, 4.8500414e-8,
              -7.220169e-11, -1.330633e-7, -0.424155,
              -0.9055898, -1.2834434e-7, -8.55035e-8,
              1.5876523e-7, 7.258963e-8, -4.021065e-9,
              -5.26491e-7, 4.0015044e-7, 5.867231e-9,
              1.02724776e-7, 0.99111766, -0.13298787]
        );
        let s_expected = DVector::from_row_slice(&[1.75729548, 1.30831209, 1.00000001]);
        let vt_expected = DMatrix::from_column_slice(3, 7, &[2.1513551e-7, -0.55492747, -3.685709e-7, -0.17350304, -3.585592e-8, -1.3154931e-8, -0.9848333, -2.6985072e-7, 1.5828846e-8, 1.7054845e-7, -0.4614543, 2.2088352e-7, 3.091991e-7, -0.69218147, 3.76167e-7, -1.6336292e-9, 3.854299e-8, -0.13298778, 2.9922877e-8, 1.6760511e-7, 0.9911177]);

        assert_relative_eq!(u, u_expected, epsilon=1e-5);
        assert_relative_eq!(singular_values, s_expected, epsilon=1e-5);
        assert_eq!(v_t, vt_expected);
    }


    #[test]
    fn test_find_range() {
        let A = get_csr();
        let mut rng = StdRng::seed_from_u64(1337);
        let Q = find_range(&A, 6, Some(1), &mut rng);
        let expected = DMatrix::from_column_slice(
            7, 6,
            &[
                -0.48018134, 0.19054835, -0.49534357, -0.49180463, -0.49180463, 0.06275115,
                -0.008419719, -0.24600098, 0.292427, 0.82924294, -0.22750038, -0.22750038,
                0.18705094, -0.1668085, 0.34559372, 0.49603245, -0.22479154, 0.08883309,
                0.08883309, 0.7322999, -0.17871745, -0.088009864, -0.75435144, 0.06704224,
                -0.09810532, -0.098105155, 0.6229459, 0.10678491, -0.68190473, -0.01649493,
                -0.09763475, 0.38894016, 0.38894016, 0.09589665, -0.46201357, -0.341665,
                0.25062254, 0.049319506, 0.1938594, 0.1938594, 0.16599235, 0.8457569,
            ],
        );
        assert_relative_eq!(Q, expected, epsilon=1e-6);
    }

}
