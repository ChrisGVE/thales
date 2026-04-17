//! Symbolic substitution solver for 2-equation, 2-variable nonlinear systems.
//!
//! Solves systems of the form `{ eq1, eq2 }` in two unknowns by:
//! 1. Isolating one variable from one equation (symbolic rearrangement)
//! 2. Substituting that expression into the other equation
//! 3. Solving the resulting single-variable equation
//! 4. Back-substituting to recover the paired solutions

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::resolution_path::ResolutionPathBuilder;
use crate::solver::symbolic_isolation::symbolic_isolate;
use crate::solver::types::SolverError;
use crate::solver::{QuadraticSolver, SmartSolver, Solution, Solver};

/// Replace every occurrence of `var_name` inside `expr` with `replacement`.
fn substitute_expr(expr: &Expression, var_name: &str, replacement: &Expression) -> Expression {
    match expr {
        Expression::Variable(v) if v.name == var_name => replacement.clone(),
        Expression::Variable(_)
        | Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Rational(_)
        | Expression::Complex(_)
        | Expression::Constant(_) => expr.clone(),
        Expression::Binary(op, left, right) => Expression::Binary(
            *op,
            Box::new(substitute_expr(left, var_name, replacement)),
            Box::new(substitute_expr(right, var_name, replacement)),
        ),
        Expression::Unary(op, inner) => {
            Expression::Unary(*op, Box::new(substitute_expr(inner, var_name, replacement)))
        }
        Expression::Power(base, exp) => Expression::Power(
            Box::new(substitute_expr(base, var_name, replacement)),
            Box::new(substitute_expr(exp, var_name, replacement)),
        ),
        Expression::Function(func, args) => Expression::Function(
            func.clone(),
            args.iter()
                .map(|arg| substitute_expr(arg, var_name, replacement))
                .collect(),
        ),
    }
}

/// Expand an expression by distributing Mul over Add/Sub.
///
/// Converts forms like `x*(5-x)` into `5*x - x*x` so that polynomial
/// coefficient extractors can recognise the degree of the expression.
fn expand(expr: &Expression) -> Expression {
    match expr {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            let l = expand(left);
            let r = expand(right);
            match (&l, &r) {
                // a * (b + c) → a*b + a*c
                (_, Expression::Binary(BinaryOp::Add, rb, rc)) => Expression::Binary(
                    BinaryOp::Add,
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(l.clone()),
                        rb.clone(),
                    ))),
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(l.clone()),
                        rc.clone(),
                    ))),
                ),
                // a * (b - c) → a*b - a*c
                (_, Expression::Binary(BinaryOp::Sub, rb, rc)) => Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(l.clone()),
                        rb.clone(),
                    ))),
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        Box::new(l.clone()),
                        rc.clone(),
                    ))),
                ),
                // (a + b) * c → a*c + b*c
                (Expression::Binary(BinaryOp::Add, la, lb), _) => Expression::Binary(
                    BinaryOp::Add,
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        la.clone(),
                        Box::new(r.clone()),
                    ))),
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        lb.clone(),
                        Box::new(r.clone()),
                    ))),
                ),
                // (a - b) * c → a*c - b*c
                (Expression::Binary(BinaryOp::Sub, la, lb), _) => Expression::Binary(
                    BinaryOp::Sub,
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        la.clone(),
                        Box::new(r.clone()),
                    ))),
                    Box::new(expand(&Expression::Binary(
                        BinaryOp::Mul,
                        lb.clone(),
                        Box::new(r.clone()),
                    ))),
                ),
                _ => Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r)),
            }
        }
        Expression::Binary(op, left, right) => {
            Expression::Binary(*op, Box::new(expand(left)), Box::new(expand(right)))
        }
        Expression::Unary(op, inner) => Expression::Unary(*op, Box::new(expand(inner))),
        Expression::Power(base, exp) => {
            Expression::Power(Box::new(expand(base)), Box::new(expand(exp)))
        }
        Expression::Function(func, args) => {
            Expression::Function(func.clone(), args.iter().map(expand).collect())
        }
        _ => expr.clone(),
    }
}

/// Collect all solution expressions from a [`Solution`] value.
fn expressions_from_solution(solution: Solution) -> Vec<Expression> {
    match solution {
        Solution::Unique(expr) => vec![expr],
        Solution::Multiple(exprs) => exprs,
        Solution::Parametric { expression, .. } => vec![expression],
        Solution::None | Solution::Infinite => vec![],
    }
}

/// Symbolic substitution solver for 2-equation, 2-variable nonlinear systems.
///
/// Solves a system `{ eq1, eq2 }` symbolically by isolating one variable from
/// one equation, substituting into the other, and back-substituting each result.
///
/// # Example
///
/// ```
/// use thales::equation_system::SubstitutionSolver;
/// use thales::ast::{Equation, Expression, BinaryOp, Variable};
///
/// // Solve: x + y = 5, x * y = 6  →  (2, 3) and (3, 2)
/// let x = Variable::new("x");
/// let y = Variable::new("y");
/// let xv = Expression::Variable(x.clone());
/// let yv = Expression::Variable(y.clone());
///
/// // eq1: x + y = 5
/// let eq1 = Equation::new(
///     "eq1",
///     Expression::Binary(BinaryOp::Add, Box::new(xv.clone()), Box::new(yv.clone())),
///     Expression::Integer(5),
/// );
/// // eq2: x * y = 6
/// let eq2 = Equation::new(
///     "eq2",
///     Expression::Binary(BinaryOp::Mul, Box::new(xv.clone()), Box::new(yv.clone())),
///     Expression::Integer(6),
/// );
///
/// let solver = SubstitutionSolver::new();
/// let solutions = solver.solve(&eq1, &eq2, &x, &y).unwrap();
/// assert_eq!(solutions.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct SubstitutionSolver {
    smart_solver: SmartSolver,
    quadratic_solver: QuadraticSolver,
}

impl SubstitutionSolver {
    /// Create a new `SubstitutionSolver`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solve a 2-equation, 2-variable system via substitution.
    ///
    /// Returns all `(var1_value, var2_value)` solution pairs found.
    /// Returns `SolverError::CannotSolve` when no substitution strategy succeeds.
    pub fn solve(
        &self,
        eq1: &Equation,
        eq2: &Equation,
        var1: &Variable,
        var2: &Variable,
    ) -> Result<Vec<(Expression, Expression)>, SolverError> {
        // Try all four (source_eq, isolated_var, target_eq, remaining_var) combos.
        let combos: [(&Equation, &Variable, &Equation, &Variable); 4] = [
            (eq1, var1, eq2, var2),
            (eq1, var2, eq2, var1),
            (eq2, var1, eq1, var2),
            (eq2, var2, eq1, var1),
        ];

        let mut all_pairs: Vec<(Expression, Expression)> = Vec::new();
        let mut last_error =
            SolverError::CannotSolve("no substitution strategy succeeded".to_string());

        let empty_map = std::collections::HashMap::<String, f64>::new();
        for (source, iso_var, target, remaining_var) in combos {
            match self.try_substitution(source, iso_var, target, remaining_var, var1) {
                Ok(pairs) => {
                    for pair in pairs {
                        let pv1 = pair.0.evaluate(&empty_map);
                        let pv2 = pair.1.evaluate(&empty_map);

                        // Verify the pair satisfies both original equations.
                        let valid = self.verify_pair(eq1, eq2, var1, var2, pv1, pv2);
                        if !valid {
                            continue;
                        }

                        let duplicate = all_pairs.iter().any(|(a, b)| {
                            if let (Some(p1), Some(p2), Some(a1), Some(b1)) =
                                (pv1, pv2, a.evaluate(&empty_map), b.evaluate(&empty_map))
                            {
                                (p1 - a1).abs() < 1e-9 && (p2 - b1).abs() < 1e-9
                            } else {
                                false
                            }
                        });
                        if !duplicate {
                            all_pairs.push(pair);
                        }
                    }
                }
                Err(e) => last_error = e,
            }
        }

        if all_pairs.is_empty() {
            Err(last_error)
        } else {
            Ok(all_pairs)
        }
    }

    /// Attempt one substitution strategy.
    ///
    /// Isolates `iso_var` from `source`, substitutes into `target`, solves for
    /// `remaining_var`, then back-substitutes to recover `iso_var` values.
    /// The `var1` parameter is used to normalise the pair order to `(var1, var2)`.
    fn try_substitution(
        &self,
        source: &Equation,
        iso_var: &Variable,
        target: &Equation,
        remaining_var: &Variable,
        var1: &Variable,
    ) -> Result<Vec<(Expression, Expression)>, SolverError> {
        // Step 1: isolate iso_var from source equation.
        let path = ResolutionPathBuilder::new(source.left.clone());
        let lhs_arc = crate::numeric::compile::compile(&source.left);
        let rhs_arc = crate::numeric::compile::compile(&source.right);
        let (iso_expr, _) = symbolic_isolate(&lhs_arc, &rhs_arc, iso_var, path)?;

        // Step 2: substitute iso_var = iso_expr into both sides of target.
        let new_left = substitute_expr(&target.left, &iso_var.name, &iso_expr);
        let new_right = substitute_expr(&target.right, &iso_var.name, &iso_expr);

        // Normalise to `lhs - rhs = 0` and expand products so that polynomial
        // coefficient extractors can detect the degree (e.g. x*(5-x) → 5x - x²).
        let raw = Expression::Binary(BinaryOp::Sub, Box::new(new_left), Box::new(new_right));
        let combined = expand(&raw).simplify();
        let substituted = Equation::new("substituted", combined, Expression::Integer(0));

        // Step 3: collect solutions from both SmartSolver and QuadraticSolver,
        // then deduplicate by numeric value (within 1e-9 tolerance).
        let mut remaining_values: Vec<Expression> = Vec::new();

        // QuadraticSolver first: it finds both roots when the form is recognised.
        if let Ok((sol, _)) = self.quadratic_solver.solve(&substituted, remaining_var) {
            remaining_values.extend(expressions_from_solution(sol));
        }

        // SmartSolver as fallback/supplement for non-quadratic forms.
        match self.smart_solver.solve(&substituted, remaining_var) {
            Ok((sol, _)) => {
                for expr in expressions_from_solution(sol) {
                    let v = expr.evaluate(&std::collections::HashMap::new());
                    let duplicate = remaining_values.iter().any(|e| {
                        matches!(
                            (v, e.evaluate(&std::collections::HashMap::new())),
                            (Some(a), Some(b)) if (a - b).abs() < 1e-9
                        )
                    });
                    if !duplicate {
                        remaining_values.push(expr);
                    }
                }
            }
            Err(e) if remaining_values.is_empty() => return Err(e),
            Err(_) => {}
        }

        if remaining_values.is_empty() {
            return Ok(vec![]);
        }

        // Step 4: back-substitute each remaining_var value to get iso_var value.
        let mut pairs = Vec::new();
        for rem_val in remaining_values {
            let back_expr = substitute_expr(&iso_expr, &remaining_var.name, &rem_val);
            let iso_val = back_expr.simplify();
            let rem_val = rem_val.simplify();

            // Normalise to (var1_value, var2_value) order.
            let pair = if iso_var.name == var1.name {
                (iso_val, rem_val)
            } else {
                (rem_val, iso_val)
            };
            pairs.push(pair);
        }

        Ok(pairs)
    }

    /// Verify a candidate `(var1_val, var2_val)` pair satisfies both equations.
    ///
    /// Substitutes numeric values into each equation and checks that
    /// `lhs - rhs ≈ 0` within a tolerance of 1e-6.
    fn verify_pair(
        &self,
        eq1: &Equation,
        eq2: &Equation,
        var1: &Variable,
        var2: &Variable,
        v1: Option<f64>,
        v2: Option<f64>,
    ) -> bool {
        let (Some(val1), Some(val2)) = (v1, v2) else {
            return false;
        };
        let mut map = std::collections::HashMap::new();
        map.insert(var1.name.clone(), val1);
        map.insert(var2.name.clone(), val2);

        let check = |eq: &Equation| {
            let lhs = eq.left.evaluate(&map);
            let rhs = eq.right.evaluate(&map);
            matches!((lhs, rhs), (Some(l), Some(r)) if (l - r).abs() < 1e-6)
        };

        check(eq1) && check(eq2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Expression, Variable};

    fn var(name: &str) -> Variable {
        Variable::new(name)
    }

    fn vexpr(name: &str) -> Expression {
        Expression::Variable(var(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }

    fn mul(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }

    fn sub(a: Expression, b: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    /// Evaluate an expression with no free variables (should return a numeric value).
    fn eval(expr: &Expression) -> f64 {
        expr.evaluate(&std::collections::HashMap::new())
            .expect("expected numeric evaluation")
    }

    /// Sort solution pairs by their first component for stable comparison.
    fn sorted_pairs(mut pairs: Vec<(Expression, Expression)>) -> Vec<(f64, f64)> {
        let mut numeric: Vec<(f64, f64)> =
            pairs.drain(..).map(|(a, b)| (eval(&a), eval(&b))).collect();
        numeric.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        numeric
    }

    #[test]
    fn test_linear_plus_product_system() {
        // x + y = 5, x * y = 6  →  (2, 3) and (3, 2)
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", add(vexpr("x"), vexpr("y")), int(5));
        let eq2 = Equation::new("eq2", mul(vexpr("x"), vexpr("y")), int(6));

        let solver = SubstitutionSolver::new();
        let solutions = solver.solve(&eq1, &eq2, &x, &y).unwrap();
        assert!(!solutions.is_empty(), "expected at least one solution");

        let pairs = sorted_pairs(solutions);
        // Expect (2.0, 3.0) and (3.0, 2.0) in some order
        assert_eq!(pairs.len(), 2);
        assert!((pairs[0].0 - 2.0).abs() < 1e-9);
        assert!((pairs[0].1 - 3.0).abs() < 1e-9);
        assert!((pairs[1].0 - 3.0).abs() < 1e-9);
        assert!((pairs[1].1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_quadratic_plus_linear_system() {
        // x² + y = 5, x + y = 3
        // Substituting y = 3 - x: x² + (3-x) = 5 → x² - x - 2 = 0
        // Roots: x = 2 (y=1) and x = -1 (y=4)
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", add(pow(vexpr("x"), int(2)), vexpr("y")), int(5));
        let eq2 = Equation::new("eq2", add(vexpr("x"), vexpr("y")), int(3));

        let solver = SubstitutionSolver::new();
        let solutions = solver.solve(&eq1, &eq2, &x, &y).unwrap();
        assert!(!solutions.is_empty());

        let pairs = sorted_pairs(solutions);
        // Sorted by x: x = -1 (y=4) then x = 2 (y=1)
        assert_eq!(pairs.len(), 2);
        assert!((pairs[0].0 - (-1.0)).abs() < 1e-9, "x0 = {:?}", pairs[0].0);
        assert!((pairs[0].1 - 4.0).abs() < 1e-9, "y0 = {:?}", pairs[0].1);
        assert!((pairs[1].0 - 2.0).abs() < 1e-9, "x1 = {:?}", pairs[1].0);
        assert!((pairs[1].1 - 1.0).abs() < 1e-9, "y1 = {:?}", pairs[1].1);
    }

    #[test]
    fn test_no_isolation_possible_returns_error() {
        // Two equations neither of whose variables can be isolated symbolically.
        // e.g. x*y = 1 and x*y = 2 — substitution should fail (no solution or cannot isolate).
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", mul(vexpr("x"), vexpr("y")), int(1));
        let eq2 = Equation::new("eq2", mul(vexpr("x"), vexpr("y")), int(2));

        let solver = SubstitutionSolver::new();
        let result = solver.solve(&eq1, &eq2, &x, &y);
        // Either an error or empty solutions (inconsistent system)
        match result {
            Err(_) => {}
            Ok(pairs) => assert!(
                pairs.is_empty(),
                "expected no solutions, got {:?}",
                pairs.len()
            ),
        }
    }

    #[test]
    fn test_back_substitution_correctness() {
        // x - y = 1, x + y = 7  →  x = 4, y = 3
        let x = var("x");
        let y = var("y");
        let eq1 = Equation::new("eq1", sub(vexpr("x"), vexpr("y")), int(1));
        let eq2 = Equation::new("eq2", add(vexpr("x"), vexpr("y")), int(7));

        let solver = SubstitutionSolver::new();
        let solutions = solver.solve(&eq1, &eq2, &x, &y).unwrap();
        assert_eq!(solutions.len(), 1);

        // The pair ordering is (iso_var_value, remaining_var_value).
        // We know x=4 and y=3; verify by evaluating both components.
        let values: Vec<(f64, f64)> = solutions.iter().map(|(a, b)| (eval(a), eval(b))).collect();
        let has_4_3 = values
            .iter()
            .any(|&(a, b)| (a - 4.0).abs() < 1e-9 && (b - 3.0).abs() < 1e-9);
        let has_3_4 = values
            .iter()
            .any(|&(a, b)| (a - 3.0).abs() < 1e-9 && (b - 4.0).abs() < 1e-9);
        assert!(
            has_4_3 || has_3_4,
            "expected (4,3) or (3,4), got {:?}",
            values
        );
    }
}
