//! Trigonometric equation patterns.
//!
//! Operates on pre-compiled `Arc<Expr>` sides. Recognised families:
//!
//! - `sin(x) = a` → `x = asin(a)`
//! - `cos(x) = a` → `x = acos(a)`
//! - `tan(x) = a` → `x = atan(a)`
//! - `sin(c·x) = a` → `x = asin(a) / c`
//! - `c · sin(x) = a` → `x = asin(a / c)`

use std::sync::Arc;

use crate::ast::Variable;
use crate::numeric::{normalize, BigRational, Expr, FuncId, MulNode, SymbolId};
use crate::resolution_path::{Operation, ResolutionPathBuilder, StepAnnotation};

use super::super::coeff::extract_coefficient;
use super::super::helpers::contains_symbol;
use super::detection::{eval_constant, validate_trig_domain};

/// Outcome of a successful trig match: isolated solution plus the
/// numeric value the inverse function is applied to (for domain
/// validation).
struct Match {
    solution: Arc<Expr>,
    applied_value: f64,
    inverse: FuncId,
}

/// Top-level trig dispatcher. Returns the isolated variable expression
/// and the builder with the inverse-function step appended. The input
/// builder is consumed on success; on miss it is returned in the `Err`
/// arm so the caller can thread it to the next family.
pub(super) fn solve_trig_equation(
    lhs: &Arc<Expr>,
    rhs: &Arc<Expr>,
    var: SymbolId,
    variable: &Variable,
    path: ResolutionPathBuilder,
) -> Result<(Arc<Expr>, ResolutionPathBuilder), ResolutionPathBuilder> {
    let families = [
        (FuncId::Sin, FuncId::Asin),
        (FuncId::Cos, FuncId::Acos),
        (FuncId::Tan, FuncId::Atan),
    ];
    for (func, inverse) in families {
        for (left, right) in [(lhs, rhs), (rhs, lhs)] {
            if let Some(m) = match_trig(left, right, var, func, inverse) {
                if matches!(inverse, FuncId::Asin | FuncId::Acos)
                    && validate_trig_domain(m.applied_value, inverse).is_err()
                {
                    return Err(path);
                }
                let path = append_step(path, m.inverse, &m.solution, variable);
                return Ok((m.solution, path));
            }
        }
    }
    Err(path)
}

fn match_trig(
    left: &Arc<Expr>,
    right: &Arc<Expr>,
    var: SymbolId,
    func: FuncId,
    inverse: FuncId,
) -> Option<Match> {
    if contains_symbol(right, var) {
        return None;
    }
    let rhs_value = eval_constant(right);

    // Pattern: Func(func, [arg]) = right.
    if let Expr::Func(f, args) = left.as_ref() {
        if *f == func && args.len() == 1 {
            let solution = isolate_after_func(&args[0], right, var, inverse)?;
            let applied_value = rhs_value.unwrap_or(0.0);
            return Some(Match {
                solution,
                applied_value,
                inverse,
            });
        }
    }

    // Pattern: c·Func(func, [arg_with_var]) = right.
    if let Expr::Mul(node) = left.as_ref() {
        if let Some((arg, coeff)) = split_mul_of_func(node, func, var) {
            // right / coeff — work in Arc<Expr> so downstream evaluation
            // retains exact rational division when coeff is rational.
            let coeff_arc = Arc::new(Expr::Rational(coeff.clone()));
            let divided = normalize::div(Arc::clone(right), coeff_arc);
            let applied_value = eval_constant(&divided).unwrap_or(0.0);
            let inverse_applied = Expr::func(inverse, vec![divided]);
            let solution = apply_linear_inverse(&arg, inverse_applied, var)?;
            return Some(Match {
                solution,
                applied_value,
                inverse,
            });
        }
    }

    None
}

/// Given `func(arg) = right`, produce the solution `var = …` when
/// `arg` is `Symbol(var)` or a purely multiplicative `coeff·var`.
fn isolate_after_func(
    arg: &Arc<Expr>,
    right: &Arc<Expr>,
    var: SymbolId,
    inverse: FuncId,
) -> Option<Arc<Expr>> {
    if matches!(arg.as_ref(), Expr::Symbol(s) if *s == var) {
        return Some(Expr::func(inverse, vec![Arc::clone(right)]));
    }
    let coeff = extract_coefficient(arg, var).ok()?;
    let inverse_applied = Expr::func(inverse, vec![Arc::clone(right)]);
    Some(normalize::div(
        inverse_applied,
        Arc::new(Expr::Rational(coeff)),
    ))
}

/// Given `arg = inverse_applied` with `arg` being `Symbol(var)` or
/// `coeff·var`, produce `var = inverse_applied [/ coeff]`.
fn apply_linear_inverse(
    arg: &Arc<Expr>,
    inverse_applied: Arc<Expr>,
    var: SymbolId,
) -> Option<Arc<Expr>> {
    if matches!(arg.as_ref(), Expr::Symbol(s) if *s == var) {
        return Some(inverse_applied);
    }
    let coeff = extract_coefficient(arg, var).ok()?;
    Some(normalize::div(
        inverse_applied,
        Arc::new(Expr::Rational(coeff)),
    ))
}

/// Decompose a `MulNode` that represents `c · func(inner)` where `inner`
/// contains the target variable. Returns `(inner, c)`. Other factors
/// must be variable-free (they are already folded into `node.coeff`).
fn split_mul_of_func(
    node: &MulNode,
    func: FuncId,
    var: SymbolId,
) -> Option<(Arc<Expr>, BigRational)> {
    let mut inner: Option<Arc<Expr>> = None;
    for (base, exp) in &node.factors {
        let base_has = contains_symbol(base, var);
        let exp_has = contains_symbol(exp, var);
        if !base_has && !exp_has {
            continue;
        }
        if exp_has || !matches!(exp.as_ref(), Expr::Integer(n) if n.to_i64() == Some(1)) {
            return None;
        }
        if let Expr::Func(f, inner_args) = base.as_ref() {
            if *f == func && inner_args.len() == 1 {
                if inner.is_some() {
                    return None;
                }
                inner = Some(Arc::clone(&inner_args[0]));
                continue;
            }
        }
        return None;
    }
    Some((inner?, node.coeff.clone()))
}

fn append_step(
    path: ResolutionPathBuilder,
    inverse: FuncId,
    solution: &Arc<Expr>,
    variable: &Variable,
) -> ResolutionPathBuilder {
    let name = match inverse {
        FuncId::Asin => "asin",
        FuncId::Acos => "acos",
        FuncId::Atan => "atan",
        _ => "inverse",
    };
    let inverse_word = match inverse {
        FuncId::Asin => "arcsine",
        FuncId::Acos => "arccosine",
        FuncId::Atan => "arctangent",
        _ => "inverse",
    };
    let solution_expr = crate::numeric::compile::decompile(solution);
    let original = match inverse {
        FuncId::Asin => "sin",
        FuncId::Acos => "cos",
        FuncId::Atan => "tan",
        _ => "func",
    };
    path.annotated_step(
        Operation::ApplyFunction(name.to_string()),
        format!(
            "Apply {} to solve {}({}) = value",
            inverse_word, original, variable
        ),
        solution_expr,
        StepAnnotation::transcendental("Inverse Trigonometric Function"),
    )
}
