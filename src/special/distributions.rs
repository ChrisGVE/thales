//! Distribution functions: Heaviside step and Dirac delta.
//!
//! Provides numeric evaluation with derivation steps.
//! Symbolic arguments return unevaluated function expressions.

use crate::ast::{Expression, Function};
use crate::special::{SpecialFunctionError, SpecialFunctionResult};

/// Extract a finite f64 from a numeric Expression, or None for symbolic.
fn numeric_val(x: &Expression) -> Option<f64> {
    match x {
        Expression::Integer(n) => Some(*n as f64),
        Expression::Float(f) => Some(*f),
        Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
        _ => None,
    }
}

/// Format an expression compactly for step text.
fn fmt_expr(x: &Expression) -> String {
    match x {
        Expression::Integer(n) => n.to_string(),
        Expression::Float(f) => format!("{}", f),
        Expression::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expression::Variable(v) => v.name.clone(),
        _ => format!("{:?}", x),
    }
}

/// Compute the Heaviside step function H(x).
///
/// ```text
/// H(x < 0) = 0
/// H(0)     = 0.5  (symmetric convention)
/// H(x > 0) = 1
/// ```
///
/// For symbolic arguments, returns an unevaluated `Heaviside(x)` expression.
#[must_use = "computing special functions returns a result that should be used"]
pub fn heaviside(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!(
        "Computing Heaviside step function: H({})",
        fmt_expr(x)
    ));
    steps.push("H(x<0)=0, H(0)=0.5 (symmetric convention), H(x>0)=1".to_string());

    match numeric_val(x) {
        None => {
            steps.push("Argument is symbolic — returning unevaluated H(x)".to_string());
            Ok(SpecialFunctionResult::new(
                Expression::Function(Function::Heaviside, vec![x.clone()]),
                None,
                steps,
            ))
        }
        Some(v) if v < 0.0 => {
            steps.push(format!("{} < 0  ⟹  H({}) = 0", v, v));
            Ok(SpecialFunctionResult::new(
                Expression::Integer(0),
                Some(0.0),
                steps,
            ))
        }
        Some(v) if v == 0.0 => {
            steps.push("x = 0  ⟹  H(0) = 0.5 (symmetric convention)".to_string());
            Ok(SpecialFunctionResult::new(
                Expression::Float(0.5),
                Some(0.5),
                steps,
            ))
        }
        Some(v) => {
            steps.push(format!("{} > 0  ⟹  H({}) = 1", v, v));
            Ok(SpecialFunctionResult::new(
                Expression::Integer(1),
                Some(1.0),
                steps,
            ))
        }
    }
}

/// Compute the Dirac delta distribution δ(x).
///
/// ```text
/// δ(x ≠ 0) = 0
/// δ(0)     = symbolic (unevaluated — not a real number)
/// ```
///
/// For x = 0 or symbolic arguments, returns an unevaluated `DiracDelta(x)` expression.
#[must_use = "computing special functions returns a result that should be used"]
pub fn dirac_delta(x: &Expression) -> Result<SpecialFunctionResult, SpecialFunctionError> {
    let mut steps = Vec::new();
    steps.push(format!("Computing Dirac delta: δ({})", fmt_expr(x)));
    steps.push("δ(x≠0) = 0; δ(0) is a distribution (returned symbolic)".to_string());

    match numeric_val(x) {
        None => {
            steps.push("Argument is symbolic — returning unevaluated δ(x)".to_string());
            Ok(SpecialFunctionResult::new(
                Expression::Function(Function::DiracDelta, vec![x.clone()]),
                None,
                steps,
            ))
        }
        Some(v) if v != 0.0 => {
            steps.push(format!("{} ≠ 0  ⟹  δ({}) = 0", v, v));
            Ok(SpecialFunctionResult::new(
                Expression::Integer(0),
                Some(0.0),
                steps,
            ))
        }
        Some(_) => {
            // x == 0: δ(0) is not a real number — return unevaluated
            steps.push("x = 0: δ(0) is not representable as a real number".to_string());
            steps.push("Returning unevaluated δ(0)".to_string());
            Ok(SpecialFunctionResult::new(
                Expression::Function(Function::DiracDelta, vec![x.clone()]),
                None,
                steps,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heaviside_negative() {
        let result = heaviside(&Expression::Integer(-1)).unwrap();
        assert_eq!(result.numeric_value, Some(0.0));
    }

    #[test]
    fn test_heaviside_zero() {
        let result = heaviside(&Expression::Integer(0)).unwrap();
        assert_eq!(result.numeric_value, Some(0.5));
    }

    #[test]
    fn test_heaviside_positive() {
        let result = heaviside(&Expression::Integer(1)).unwrap();
        assert_eq!(result.numeric_value, Some(1.0));
    }

    #[test]
    fn test_heaviside_float_negative() {
        let result = heaviside(&Expression::Float(-2.5)).unwrap();
        assert_eq!(result.numeric_value, Some(0.0));
    }

    #[test]
    fn test_heaviside_float_positive() {
        let result = heaviside(&Expression::Float(3.7)).unwrap();
        assert_eq!(result.numeric_value, Some(1.0));
    }

    #[test]
    fn test_heaviside_symbolic() {
        use crate::ast::Variable;
        let x = Expression::Variable(Variable::new("x"));
        let result = heaviside(&x).unwrap();
        assert!(result.numeric_value.is_none());
    }

    #[test]
    fn test_dirac_nonzero_positive() {
        let result = dirac_delta(&Expression::Integer(1)).unwrap();
        assert_eq!(result.numeric_value, Some(0.0));
    }

    #[test]
    fn test_dirac_nonzero_negative() {
        let result = dirac_delta(&Expression::Integer(-1)).unwrap();
        assert_eq!(result.numeric_value, Some(0.0));
    }

    #[test]
    fn test_dirac_zero_is_symbolic() {
        let result = dirac_delta(&Expression::Integer(0)).unwrap();
        assert!(result.numeric_value.is_none());
    }

    #[test]
    fn test_dirac_symbolic() {
        use crate::ast::Variable;
        let x = Expression::Variable(Variable::new("x"));
        let result = dirac_delta(&x).unwrap();
        assert!(result.numeric_value.is_none());
    }

    #[test]
    fn test_derivation_steps_populated() {
        let h = heaviside(&Expression::Integer(1)).unwrap();
        assert!(!h.derivation_steps.is_empty());
        let d = dirac_delta(&Expression::Integer(1)).unwrap();
        assert!(!d.derivation_steps.is_empty());
    }
}
