#[cfg(test)]
mod tests {
    use super::super::extract::collect_poly_coeffs;
    use super::super::*;
    use crate::numeric::expr::Expr;

    fn int_entry(n: i64) -> Arc<Expr> {
        Expr::int(n)
    }

    // ── characteristic_polynomial ────────────────────────────────────────────

    /// Characteristic polynomial of [[1,2],[3,4]] is λ²−5λ−2.
    #[test]
    fn test_char_poly_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2)],
            vec![int_entry(3), int_entry(4)],
        ];
        let lambda = SymbolId::intern("l");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        // Verify by checking eigenvalues indirectly through the numeric path
        let lam_expr = Arc::new(Expr::Symbol(lambda));
        let n = 2;
        let mut poly_coeffs = vec![BigRational::zero(); n + 1];
        let one = BigRational::one();
        let ok = collect_poly_coeffs(&cp, lambda, 0, &one, &mut poly_coeffs);
        assert!(ok.is_some(), "char poly must be extractable");
        // λ²−5λ−2 → coeffs = [-2, -5, 1]
        let coeffs_i64: Vec<i64> = poly_coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // constant = -2, linear = -5, quadratic = 1
        assert_eq!(coeffs_i64[0], -2, "constant term");
        assert_eq!(coeffs_i64[1], -5, "linear term");
        assert_eq!(coeffs_i64[2], 1, "quadratic term");
        drop(lam_expr);
    }

    /// Characteristic polynomial of identity 2×2 is (λ−1)² = λ²−2λ+1.
    #[test]
    fn test_char_poly_identity_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let lambda = SymbolId::intern("mu");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        let mut coeffs = vec![BigRational::zero(); 3];
        let ok = collect_poly_coeffs(&cp, lambda, 0, &BigRational::one(), &mut coeffs);
        assert!(ok.is_some());
        let ci: Vec<i64> = coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // (λ−1)² = λ²−2λ+1 → [1, -2, 1]
        assert_eq!(ci[0], 1);
        assert_eq!(ci[1], -2);
        assert_eq!(ci[2], 1);
    }

    /// Characteristic polynomial of [[5]] is λ−5.
    #[test]
    fn test_char_poly_1x1() {
        let m: ExprMatrix = vec![vec![int_entry(5)]];
        let lambda = SymbolId::intern("nu");
        let cp = characteristic_polynomial(&m, lambda).unwrap();
        let mut coeffs = vec![BigRational::zero(); 2];
        collect_poly_coeffs(&cp, lambda, 0, &BigRational::one(), &mut coeffs).unwrap();
        let ci: Vec<i64> = coeffs
            .iter()
            .map(|r| r.numer().to_i64().unwrap_or(0))
            .collect();
        // 5 − λ → [5, -1]
        assert_eq!(ci[0], 5);
        assert_eq!(ci[1], -1);
    }

    // ── eigenvalues: numeric matrices ────────────────────────────────────────

    /// Eigenvalues of [[1,0],[0,1]] are {1} with multiplicity 2.
    #[test]
    fn test_eigenvalues_identity_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(evs) => {
                assert_eq!(evs.len(), 1, "one distinct eigenvalue");
                assert_eq!(evs[0].1, 2, "multiplicity 2");
                match evs[0].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(1)),
                    _ => panic!("eigenvalue should be integer 1"),
                }
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric result"),
        }
    }

    /// Eigenvalues of [[2,0],[0,3]] are {2, 3} each with multiplicity 1.
    #[test]
    fn test_eigenvalues_diagonal_2x2() {
        let m: ExprMatrix = vec![
            vec![int_entry(2), int_entry(0)],
            vec![int_entry(0), int_entry(3)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(mut evs) => {
                assert_eq!(evs.len(), 2);
                evs.sort_by_key(|(e, _)| match e.as_ref() {
                    Expr::Integer(n) => n.to_i64().unwrap_or(0),
                    _ => 0,
                });
                assert_eq!(evs[0].1, 1);
                assert_eq!(evs[1].1, 1);
                match evs[0].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(2)),
                    _ => panic!("expected integer 2"),
                }
                match evs[1].0.as_ref() {
                    Expr::Integer(n) => assert_eq!(n.to_i64(), Some(3)),
                    _ => panic!("expected integer 3"),
                }
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric"),
        }
    }

    /// [[1,2],[3,4]]: char poly λ²−5λ−2. Roots are irrational so solver
    /// returns no numeric roots; result should be Symbolic or Numeric([]).
    #[test]
    fn test_eigenvalues_1_2_3_4() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2)],
            vec![int_entry(3), int_entry(4)],
        ];
        let result = eigenvalues(&m).unwrap();
        // The char poly is λ²−5λ−2 which has irrational roots;
        // roots_with_multiplicity returns empty for non-rational-square discriminant.
        match result {
            EigenvalueResult::Numeric(evs) => {
                // No rational roots — the solver correctly returns empty
                assert!(evs.is_empty(), "irrational roots not returned numerically");
            }
            EigenvalueResult::Symbolic(_) => {
                // Also acceptable: char poly returned as symbolic
            }
        }
    }

    /// Eigenvalues of [[3,-2],[1,0]]: char poly λ²−3λ+2 = (λ−1)(λ−2).
    #[test]
    fn test_eigenvalues_rational_roots() {
        let m: ExprMatrix = vec![
            vec![int_entry(3), int_entry(-2)],
            vec![int_entry(1), int_entry(0)],
        ];
        let result = eigenvalues(&m).unwrap();
        match result {
            EigenvalueResult::Numeric(mut evs) => {
                assert_eq!(evs.len(), 2, "two distinct eigenvalues");
                evs.sort_by_key(|(e, _)| match e.as_ref() {
                    Expr::Integer(n) => n.to_i64().unwrap_or(0),
                    _ => 0,
                });
                let vals: Vec<i64> = evs
                    .iter()
                    .filter_map(|(e, _)| match e.as_ref() {
                        Expr::Integer(n) => n.to_i64(),
                        _ => None,
                    })
                    .collect();
                assert_eq!(vals, vec![1, 2]);
            }
            EigenvalueResult::Symbolic(_) => panic!("expected numeric eigenvalues"),
        }
    }

    // ── eigenvalues: symbolic matrices ───────────────────────────────────────

    /// A matrix with a symbol entry should return a Symbolic result.
    #[test]
    fn test_eigenvalues_symbolic_matrix() {
        let a = Arc::new(Expr::Symbol(SymbolId::intern("a")));
        let m: ExprMatrix = vec![
            vec![a.clone(), int_entry(0)],
            vec![int_entry(0), int_entry(1)],
        ];
        let result = eigenvalues(&m).unwrap();
        assert!(
            matches!(result, EigenvalueResult::Symbolic(_)),
            "symbolic matrix must yield Symbolic result"
        );
    }

    // ── error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_non_square_returns_error() {
        let m: ExprMatrix = vec![
            vec![int_entry(1), int_entry(2), int_entry(3)],
            vec![int_entry(4), int_entry(5), int_entry(6)],
        ];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::NotSquare)
        ));
        assert!(matches!(eigenvalues(&m), Err(EigenError::NotSquare)));
    }

    #[test]
    fn test_empty_matrix_returns_error() {
        let m: ExprMatrix = vec![];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::Empty)
        ));
    }

    #[test]
    fn test_ragged_matrix_returns_error() {
        let m: ExprMatrix = vec![vec![int_entry(1), int_entry(2)], vec![int_entry(3)]];
        assert!(matches!(
            characteristic_polynomial(&m, SymbolId::intern("x")),
            Err(EigenError::RaggedMatrix)
        ));
    }

    // ── symbolic 2×2 eigenvalues ─────────────────────────────────────────────

    /// For the symbolic 2×2 matrix [[a, b], [c, d]], the characteristic
    /// polynomial should be expressible symbolically.
    #[test]
    fn test_symbolic_2x2_char_poly() {
        let a = Arc::new(Expr::Symbol(SymbolId::intern("sa")));
        let b = Arc::new(Expr::Symbol(SymbolId::intern("sb")));
        let c = Arc::new(Expr::Symbol(SymbolId::intern("sc")));
        let d = Arc::new(Expr::Symbol(SymbolId::intern("sd")));
        let m: ExprMatrix = vec![vec![a, b], vec![c, d]];
        let lambda = SymbolId::intern("slambda");
        // Should succeed (no error), producing a symbolic expression
        let cp = characteristic_polynomial(&m, lambda);
        assert!(cp.is_ok(), "char poly of symbolic 2×2 should not error");
    }
}
