# Randomized Singular Value Decomposition

An [nalgebra](https://github.com/dimforge/nalgebra) compatible rust crate to compute the [Randomized SVD](https://gregorygundersen.com/blog/2019/01/17/randomized-svd/) of a sparse matrix.
It exposes a struct `RSVD`, and one function: `rsvd`.

Algorithm (1): 

![Screenshot_20230915_153659](https://github.com/shur-complement/randomized-svd/assets/139090555/5ace637d-a7e7-4ba5-90a4-3c42aaeb180c)

# Example
```rust
use rand::prelude::*;
use nalgebra_sparse::CsrMatrix;

fn compute() {
  let indptr = vec![0, 2, 3, 6, 7, 8, 9, 10];
  let indices = vec![1, 2, 0, 0, 3, 4, 2, 2, 6, 5];
  let data = vec![0.5, 1., 1., 0.33333334, 0.66666669, 1., 1., 1., 1., 1.,];
  let A = CsrMatrix::try_from_csr_data(
      7,7,
      indptr,
      indices,
      data,
  ).unwrap();

  let mut rng = thread_rng();
  let RSVD{u, singular_values, v_t} = rsvd(&A, 3, Some(3), Some(1), &mut rng);
}
```

# Further Improvements

This isn't going to be the most optimized implementation, but it is easily modifiable for anyone as it is so simple.
Feel free to copy-paste any code, or submit PRs.

# Support

If you need any changes, open an issue and it will be addressed.

# References

1) [Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions
Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp](https://arxiv.org/abs/0909.4061)
