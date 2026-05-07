//! Transpose and other arithmetic operations on [`MatrixExpr`].

use std::sync::Arc;

use crate::numeric::expr::Expr;
use crate::numeric::normalize;

use super::{MatrixError, MatrixExpr, MatrixResult};

impl MatrixExpr {
    /// Compute the transpose of this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2), Expr::int(3)],
    ///     vec![Expr::int(4), Expr::int(5), Expr::int(6)],
    /// ]).unwrap();
    ///
    /// let mt = m.transpose();
    /// assert_eq!(mt.rows(), 3);
    /// assert_eq!(mt.cols(), 2);
    /// ```
    pub fn transpose(&self) -> Self {
        let elements: Vec<Vec<Arc<Expr>>> = (0..self.cols)
            .map(|j| {
                (0..self.rows)
                    .map(|i| self.elements[i][j].clone())
                    .collect()
            })
            .collect();
        Self {
            rows: self.cols,
            cols: self.rows,
            elements,
        }
    }

    /// Compute the Kronecker (tensor) product of `self` ⊗ `other`.
    ///
    /// For an m×n matrix A and a p×q matrix B, the result is an (m·p)×(n·q)
    /// matrix where block (i, j) equals `A[i][j] * B`.
    ///
    /// # Errors
    ///
    /// Returns an error if either matrix is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::MatrixExpr;
    /// use thales::numeric::expr::Expr;
    ///
    /// let a = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(0)],
    ///     vec![Expr::int(0), Expr::int(1)],
    /// ]).unwrap();
    ///
    /// let b = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let kp = a.kronecker_product(&b).unwrap();
    /// assert_eq!(kp.rows(), 4);
    /// assert_eq!(kp.cols(), 4);
    /// ```
    pub fn kronecker_product(&self, other: &MatrixExpr) -> MatrixResult<MatrixExpr> {
        if self.rows == 0 || self.cols == 0 || other.rows == 0 || other.cols == 0 {
            return Err(MatrixError::InvalidOperation(
                "Kronecker product requires non-empty matrices".to_string(),
            ));
        }

        let m = self.rows;
        let n = self.cols;
        let p = other.rows;
        let q = other.cols;

        let mut elements: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(m * p);

        for i in 0..m {
            for pi in 0..p {
                let mut row = Vec::with_capacity(n * q);
                for j in 0..n {
                    let a_ij = &self.elements[i][j];
                    for qj in 0..q {
                        row.push(normalize::mul(a_ij.clone(), other.elements[pi][qj].clone()));
                    }
                }
                elements.push(row);
            }
        }

        MatrixExpr::from_expr_elements(elements)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::matrix::MatrixExpr;
    use crate::numeric::evaluation::evaluate;
    use crate::numeric::expr::Expr;
    use crate::numeric::SymbolId;

    fn eval(e: &std::sync::Arc<Expr>) -> Option<f64> {
        let empty: HashMap<SymbolId, f64> = HashMap::new();
        evaluate(e, &empty)
    }

    // ---------------------------------------------------------------
    // test_kronecker_dimensions — (2×2) ⊗ (3×3) = 6×6
    // ---------------------------------------------------------------
    #[test]
    fn test_kronecker_dimensions() {
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(3), Expr::int(4)],
        ])
        .unwrap();

        let b = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0), Expr::int(0)],
            vec![Expr::int(0), Expr::int(1), Expr::int(0)],
            vec![Expr::int(0), Expr::int(0), Expr::int(1)],
        ])
        .unwrap();

        let kp = a.kronecker_product(&b).unwrap();
        assert_eq!(kp.rows(), 6);
        assert_eq!(kp.cols(), 6);
    }

    // ---------------------------------------------------------------
    // test_kronecker_identity — A ⊗ I₂ has expected block structure
    // ---------------------------------------------------------------
    #[test]
    fn test_kronecker_identity() {
        // A = [[2, 3], [5, 7]]
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(3)],
            vec![Expr::int(5), Expr::int(7)],
        ])
        .unwrap();

        let id2 = MatrixExpr::identity(2);

        // A ⊗ I₂ should be block-diagonal with A's scalings:
        // [[2,0, 3,0],
        //  [0,2, 0,3],
        //  [5,0, 7,0],
        //  [0,5, 0,7]]
        let kp = a.kronecker_product(&id2).unwrap();
        assert_eq!(kp.rows(), 4);
        assert_eq!(kp.cols(), 4);

        let elems = kp.elements();
        // Row 0: [2, 0, 3, 0]
        assert_eq!(eval(&elems[0][0]), Some(2.0));
        assert_eq!(eval(&elems[0][1]), Some(0.0));
        assert_eq!(eval(&elems[0][2]), Some(3.0));
        assert_eq!(eval(&elems[0][3]), Some(0.0));
        // Row 1: [0, 2, 0, 3]
        assert_eq!(eval(&elems[1][0]), Some(0.0));
        assert_eq!(eval(&elems[1][1]), Some(2.0));
        assert_eq!(eval(&elems[1][2]), Some(0.0));
        assert_eq!(eval(&elems[1][3]), Some(3.0));
    }

    // ---------------------------------------------------------------
    // test_kronecker_scalar — scalar(k) ⊗ B = k*B
    // ---------------------------------------------------------------
    #[test]
    fn test_kronecker_scalar() {
        let k = MatrixExpr::from_expr_elements(vec![vec![Expr::int(3)]]).unwrap();

        let b = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(2)],
            vec![Expr::int(4), Expr::int(5)],
        ])
        .unwrap();

        let kp = k.kronecker_product(&b).unwrap();
        assert_eq!(kp.rows(), 2);
        assert_eq!(kp.cols(), 2);

        let elems = kp.elements();
        assert_eq!(eval(&elems[0][0]), Some(3.0));
        assert_eq!(eval(&elems[0][1]), Some(6.0));
        assert_eq!(eval(&elems[1][0]), Some(12.0));
        assert_eq!(eval(&elems[1][1]), Some(15.0));
    }

    // ---------------------------------------------------------------
    // test_kronecker_trace — tr(A ⊗ B) = tr(A) × tr(B) for square matrices
    // ---------------------------------------------------------------
    #[test]
    fn test_kronecker_trace() {
        // A = [[2, 0], [0, 3]]  tr(A) = 5
        let a = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(2), Expr::int(0)],
            vec![Expr::int(0), Expr::int(3)],
        ])
        .unwrap();

        // B = [[1, 0], [0, 4]]  tr(B) = 5
        let b = MatrixExpr::from_expr_elements(vec![
            vec![Expr::int(1), Expr::int(0)],
            vec![Expr::int(0), Expr::int(4)],
        ])
        .unwrap();

        let kp = a.kronecker_product(&b).unwrap();

        let tr_kp = kp.trace().unwrap();
        // tr(A) * tr(B) = 5 * 5 = 25
        assert_eq!(eval(&tr_kp), Some(25.0));
    }

    // ---------------------------------------------------------------
    // test_kronecker_1x1 — edge case with 1×1 matrices
    // ---------------------------------------------------------------
    #[test]
    fn test_kronecker_1x1() {
        let a = MatrixExpr::from_expr_elements(vec![vec![Expr::int(7)]]).unwrap();
        let b = MatrixExpr::from_expr_elements(vec![vec![Expr::int(6)]]).unwrap();

        let kp = a.kronecker_product(&b).unwrap();
        assert_eq!(kp.rows(), 1);
        assert_eq!(kp.cols(), 1);
        assert_eq!(eval(&kp.elements()[0][0]), Some(42.0));
    }
}
