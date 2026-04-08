//! Inequality solving for linear and quadratic expressions.
//!
//! Solves single inequalities and systems of inequalities,
//! returning solution sets as intervals.

mod solver;
mod system;
mod types;

pub use solver::solve_inequality;
pub use system::solve_system;
pub use types::{Bound, Inequality, InequalityError, IntervalSolution};

#[cfg(test)]
mod tests {
    use super::system::*;
    use super::*;
    use crate::ast::{BinaryOp, Expression, Function, UnaryOp, Variable};

    fn var(name: &str) -> Expression {
        Expression::Variable(Variable::new(name))
    }

    fn int(n: i64) -> Expression {
        Expression::Integer(n)
    }

    fn add(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul(l: Expression, r: Expression) -> Expression {
        Expression::Binary(BinaryOp::Mul, Box::new(l), Box::new(r))
    }

    fn pow(base: Expression, exp: Expression) -> Expression {
        Expression::Power(Box::new(base), Box::new(exp))
    }

    #[test]
    fn test_linear_greater_than() {
        // 2x + 3 > 7  =>  2x > 4  =>  x > 2
        let lhs = add(mul(int(2), var("x")), int(3));
        let ineq = Inequality::GreaterThan(lhs, int(7));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (2, +∞)
        if let IntervalSolution::Interval {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            assert!(matches!(upper, Bound::PositiveInfinity));
            if let Bound::Value(e) = lower {
                let val = eval_constant(&e).unwrap();
                assert!((val - 2.0).abs() < 1e-10);
            } else {
                panic!("Expected Value bound");
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_quadratic_less_than() {
        // x^2 - 4 < 0  =>  -2 < x < 2
        let x_sq = pow(var("x"), int(2));
        let lhs = sub(x_sq, int(4));
        let ineq = Inequality::LessThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (-2, 2)
        if let IntervalSolution::Interval {
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let (Bound::Value(l), Bound::Value(u)) = (lower, upper) {
                let vl = eval_constant(&l).unwrap();
                let vu = eval_constant(&u).unwrap();
                assert!((vl - (-2.0)).abs() < 1e-10);
                assert!((vu - 2.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_quadratic_greater_equal() {
        // x^2 - 4 >= 0  =>  x <= -2 OR x >= 2
        let x_sq = pow(var("x"), int(2));
        let lhs = sub(x_sq, int(4));
        let ineq = Inequality::GreaterEqual(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be union: (-∞, -2] ∪ [2, +∞)
        assert!(matches!(solution, IntervalSolution::Union(_)));
    }

    #[test]
    fn test_linear_flip_sign() {
        // -x + 3 > 0  =>  x < 3
        let lhs = add(Expression::Unary(UnaryOp::Neg, Box::new(var("x"))), int(3));
        let ineq = Inequality::GreaterThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();

        // Should be (-∞, 3)
        if let IntervalSolution::Interval {
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        } = solution
        {
            assert!(matches!(lower, Bound::NegativeInfinity));
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let Bound::Value(u) = upper {
                let vu = eval_constant(&u).unwrap();
                assert!((vu - 3.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution");
        }
    }

    #[test]
    fn test_system_intersection() {
        // x > 0 AND x < 5  =>  (0, 5)
        let ineq1 = Inequality::GreaterThan(var("x"), int(0));
        let ineq2 = Inequality::LessThan(var("x"), int(5));

        let solution = solve_system(&[ineq1, ineq2], "x").unwrap();

        // Should be (0, 5)
        if let IntervalSolution::Interval {
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        } = solution
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
            if let (Bound::Value(l), Bound::Value(u)) = (lower, upper) {
                let vl = eval_constant(&l).unwrap();
                let vu = eval_constant(&u).unwrap();
                assert!((vl - 0.0).abs() < 1e-10);
                assert!((vu - 5.0).abs() < 1e-10);
            }
        } else {
            panic!("Expected Interval solution: {:?}", solution);
        }
    }

    #[test]
    fn test_no_solution() {
        // x^2 + 1 < 0 has no real solutions
        let x_sq = pow(var("x"), int(2));
        let lhs = add(x_sq, int(1));
        let ineq = Inequality::LessThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();
        assert!(matches!(solution, IntervalSolution::Empty));
    }

    #[test]
    fn test_all_reals_solution() {
        // x^2 + 1 > 0 is always true
        let x_sq = pow(var("x"), int(2));
        let lhs = add(x_sq, int(1));
        let ineq = Inequality::GreaterThan(lhs, int(0));

        let solution = solve_inequality(&ineq, "x").unwrap();
        assert!(matches!(solution, IntervalSolution::AllReals));
    }
}
