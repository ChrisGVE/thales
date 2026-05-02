//! System of inequalities solver and interval operations.

use crate::ast::Expression;
use std::collections::HashMap;

use super::solver::solve_inequality;
use super::types::{Bound, Inequality, InequalityResult, IntervalSolution};

pub fn solve_system(inequalities: &[Inequality], var: &str) -> InequalityResult {
    if inequalities.is_empty() {
        return Ok(IntervalSolution::AllReals);
    }

    // Solve each inequality
    let mut solutions = Vec::new();
    for ineq in inequalities {
        solutions.push(solve_inequality(ineq, var)?);
    }

    // Intersect all solutions
    let mut result = solutions[0].clone();
    for sol in &solutions[1..] {
        result = intersect_intervals(&result, sol);
    }

    Ok(result)
}

/// Intersect two interval solutions.
fn intersect_intervals(a: &IntervalSolution, b: &IntervalSolution) -> IntervalSolution {
    match (a, b) {
        (IntervalSolution::Empty, _) | (_, IntervalSolution::Empty) => IntervalSolution::Empty,
        (IntervalSolution::AllReals, other) | (other, IntervalSolution::AllReals) => other.clone(),
        (IntervalSolution::Union(intervals_a), _) => {
            // Intersect each interval in a with b
            let mut results = Vec::new();
            for int_a in intervals_a {
                let intersection = intersect_intervals(int_a, b);
                if !matches!(intersection, IntervalSolution::Empty) {
                    results.push(intersection);
                }
            }
            if results.is_empty() {
                IntervalSolution::Empty
            } else if results.len() == 1 {
                results.pop().unwrap()
            } else {
                IntervalSolution::Union(results)
            }
        }
        (_, IntervalSolution::Union(intervals_b)) => {
            // Intersect a with each interval in b
            let mut results = Vec::new();
            for int_b in intervals_b {
                let intersection = intersect_intervals(a, int_b);
                if !matches!(intersection, IntervalSolution::Empty) {
                    results.push(intersection);
                }
            }
            if results.is_empty() {
                IntervalSolution::Empty
            } else if results.len() == 1 {
                results.pop().unwrap()
            } else {
                IntervalSolution::Union(results)
            }
        }
        (
            IntervalSolution::Interval {
                lower: l1,
                lower_inclusive: li1,
                upper: u1,
                upper_inclusive: ui1,
            },
            IntervalSolution::Interval {
                lower: l2,
                lower_inclusive: li2,
                upper: u2,
                upper_inclusive: ui2,
            },
        ) => {
            // Find the intersection of two single intervals
            // This is a simplified implementation for common cases
            let (new_lower, new_li) = max_bound(l1, *li1, l2, *li2);
            let (new_upper, new_ui) = min_bound(u1, *ui1, u2, *ui2);

            // Check if valid interval
            if is_valid_interval(&new_lower, &new_upper) {
                IntervalSolution::Interval {
                    lower: new_lower,
                    lower_inclusive: new_li,
                    upper: new_upper,
                    upper_inclusive: new_ui,
                }
            } else {
                IntervalSolution::Empty
            }
        }
    }
}

/// Evaluate an expression with no variables.
pub(super) fn eval_constant(expr: &Expression) -> Option<f64> {
    let empty: HashMap<String, f64> = HashMap::new();
    expr.evaluate(&empty)
}

/// Get the maximum of two lower bounds.
fn max_bound(b1: &Bound, inc1: bool, b2: &Bound, inc2: bool) -> (Bound, bool) {
    match (b1, b2) {
        (Bound::NegativeInfinity, _) => (b2.clone(), inc2),
        (_, Bound::NegativeInfinity) => (b1.clone(), inc1),
        (Bound::PositiveInfinity, _) | (_, Bound::PositiveInfinity) => {
            (Bound::PositiveInfinity, false)
        }
        (Bound::Value(e1), Bound::Value(e2)) => {
            // Try to compare numerically
            match (eval_constant(e1), eval_constant(e2)) {
                (Some(v1), Some(v2)) => {
                    if v1 > v2 {
                        (b1.clone(), inc1)
                    } else if v2 > v1 {
                        (b2.clone(), inc2)
                    } else {
                        // Equal - take less inclusive
                        (b1.clone(), inc1 && inc2)
                    }
                }
                _ => (b1.clone(), inc1), // Fallback
            }
        }
    }
}

/// Get the minimum of two upper bounds.
fn min_bound(b1: &Bound, inc1: bool, b2: &Bound, inc2: bool) -> (Bound, bool) {
    match (b1, b2) {
        (Bound::PositiveInfinity, _) => (b2.clone(), inc2),
        (_, Bound::PositiveInfinity) => (b1.clone(), inc1),
        (Bound::NegativeInfinity, _) | (_, Bound::NegativeInfinity) => {
            (Bound::NegativeInfinity, false)
        }
        (Bound::Value(e1), Bound::Value(e2)) => match (eval_constant(e1), eval_constant(e2)) {
            (Some(v1), Some(v2)) => {
                if v1 < v2 {
                    (b1.clone(), inc1)
                } else if v2 < v1 {
                    (b2.clone(), inc2)
                } else {
                    (b1.clone(), inc1 && inc2)
                }
            }
            _ => (b1.clone(), inc1),
        },
    }
}

/// Check if an interval [lower, upper] is valid (lower < upper).
fn is_valid_interval(lower: &Bound, upper: &Bound) -> bool {
    match (lower, upper) {
        (Bound::NegativeInfinity, _) => true,
        (_, Bound::PositiveInfinity) => true,
        (Bound::PositiveInfinity, _) | (_, Bound::NegativeInfinity) => false,
        (Bound::Value(l), Bound::Value(u)) => {
            match (eval_constant(l), eval_constant(u)) {
                (Some(vl), Some(vu)) => vl <= vu,
                _ => true, // Assume valid if can't compare
            }
        }
    }
}
