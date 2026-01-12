//! # Equation Solving Guide
//!
//! This guide covers equation solving workflows in thales, from simple linear equations
//! to complex multi-equation systems. Learn how to choose the right solver, work with
//! different equation types, and handle solution results effectively.
//!
//! ## Quick Start: Linear Equations
//!
//! The simplest use case is solving a linear equation with [`SmartSolver`](crate::SmartSolver):
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver};
//!
//! // Parse and solve: 2x + 3 = 11
//! let equation = parse_equation("2*x + 3 = 11").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, path) = solver.solve(&equation, &"x".into()).unwrap();
//!
//! // Solution is x = 4
//! use thales::Solution;
//! if let Solution::Unique(expr) = solution {
//!     println!("x = {}", expr);  // x = 4
//! }
//! ```
//!
//! [`SmartSolver`](crate::SmartSolver) automatically detects the equation type and selects the
//! appropriate solving method. The returned [`ResolutionPath`](crate::ResolutionPath) contains
//! step-by-step solution details for educational applications.
//!
//! ## Understanding Solution Types
//!
//! The [`Solution`](crate::Solution) enum represents different solution structures:
//!
//! ### Unique Solution
//!
//! Most linear equations have a single solution:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//!
//! let eq = parse_equation("3*x = 12").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! match solution {
//!     Solution::Unique(expr) => {
//!         // expr is Expression::Integer(4)
//!         println!("Single solution: {}", expr);
//!     }
//!     _ => panic!("Expected unique solution"),
//! }
//! ```
//!
//! ### Multiple Solutions
//!
//! Quadratic and higher-degree equations can have multiple discrete solutions:
//!
//! ```rust,ignore
//! // Quadratic: x² - 4 = 0 has solutions x = 2 and x = -2
//! let eq = parse_equation("x^2 - 4 = 0").unwrap();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! match solution {
//!     Solution::Multiple(solutions) => {
//!         // solutions contains [2, -2]
//!         for s in solutions {
//!             println!("Solution: {}", s);
//!         }
//!     }
//!     _ => panic!("Expected multiple solutions"),
//! }
//! ```
//!
//! ### No Solution
//!
//! Inconsistent equations have no solution:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//!
//! let eq = parse_equation("0 = 5").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! assert!(matches!(solution, Solution::None));
//! ```
//!
//! ### Infinite Solutions
//!
//! Identity equations are satisfied by all values:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//!
//! let eq = parse_equation("x = x").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! assert!(matches!(solution, Solution::Infinite));
//! ```
//!
//! ## Quadratic Equations
//!
//! Quadratic equations (ax² + bx + c = 0) are solved using the discriminant method:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//!
//! // Solve: x² + 5x + 6 = 0
//! let eq = parse_equation("x^2 + 5*x + 6 = 0").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, path) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! // Solutions: x = -2 and x = -3
//! match solution {
//!     Solution::Multiple(solutions) => {
//!         assert_eq!(solutions.len(), 2);
//!         // solutions contain x = -2 and x = -3
//!     }
//!     _ => panic!("Expected two solutions"),
//! }
//! ```
//!
//! The discriminant determines the number of solutions:
//! - Δ > 0: Two distinct real solutions
//! - Δ = 0: One repeated solution (double root)
//! - Δ < 0: Two complex conjugate solutions
//!
//! ## Polynomial Equations
//!
//! Higher-degree polynomial equations use companion matrix methods:
//!
//! ```rust,ignore
//! // Cubic: x³ - 6x² + 11x - 6 = 0
//! let eq = parse_equation("x^3 - 6*x^2 + 11*x - 6 = 0").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! // Solutions: x = 1, x = 2, x = 3
//! match solution {
//!     Solution::Multiple(roots) => {
//!         println!("Found {} roots", roots.len());
//!     }
//!     _ => panic!("Expected multiple solutions"),
//! }
//! ```
//!
//! ## Transcendental Equations
//!
//! Equations involving trigonometric, exponential, or logarithmic functions often
//! require numerical methods:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver};
//!
//! // Solve: sin(x) = 0.5
//! let eq = parse_equation("sin(x) = 0.5").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! // Solution: x = arcsin(0.5) = π/6
//! ```
//!
//! For complex transcendental equations that can't be solved symbolically,
//! use [`SmartNumericalSolver`](crate::SmartNumericalSolver):
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartNumericalSolver, NumericalConfig};
//!
//! // Solve: e^x = x + 2 (no closed-form solution)
//! let eq = parse_equation("exp(x) = x + 2").unwrap();
//! let config = NumericalConfig::default();
//! let solver = SmartNumericalSolver::new(config);
//! let solution = solver.solve_for(&eq, "x", 0.0).unwrap();
//!
//! println!("x ≈ {}", solution.value);
//! ```
//!
//! ## Working with Variables
//!
//! Use the [`Variable`](crate::Variable) type to represent unknowns:
//!
//! ```rust,ignore
//! use thales::{Variable, Expression, BinaryOp, Equation};
//!
//! let x = Variable::new("x");
//! let y = Variable::new("y");
//!
//! // Build expression: 2*x + y
//! let expr = Expression::Binary(
//!     BinaryOp::Add,
//!     Box::new(Expression::Binary(
//!         BinaryOp::Mul,
//!         Box::new(Expression::Integer(2)),
//!         Box::new(Expression::Variable(x.clone())),
//!     )),
//!     Box::new(Expression::Variable(y)),
//! );
//!
//! // Check variable presence
//! assert!(expr.contains_variable("x"));
//! assert!(expr.contains_variable("y"));
//! ```
//!
//! ## Multi-Equation Systems
//!
//! The [`MultiEquationSolver`](crate::MultiEquationSolver) handles systems of equations by
//! building a dependency graph and solving in the correct order:
//!
//! ### Basic System Example
//!
//! ```rust,ignore
//! use thales::{EquationSystem, SystemContext, MultiEquationSolver, parse_equation};
//!
//! // Physics example: F = ma and v = u + at
//! let mut system = EquationSystem::new();
//! system.add_equation("eq1", parse_equation("F = m * a").unwrap());
//! system.add_equation("eq2", parse_equation("v = u + a * t").unwrap());
//!
//! // Provide known values and specify targets
//! let context = SystemContext::new()
//!     .with_known_value("F", 100.0)
//!     .with_known_value("m", 20.0)
//!     .with_known_value("u", 0.0)
//!     .with_known_value("t", 5.0)
//!     .with_target("a")
//!     .with_target("v");
//!
//! // Solve the system
//! let solver = MultiEquationSolver::new();
//! let solution = solver.solve(&system, &context).unwrap();
//!
//! // Extract results
//! let a = solution.get_numeric("a").unwrap();
//! let v = solution.get_numeric("v").unwrap();
//!
//! assert!((a - 5.0).abs() < 1e-10);   // a = 5.0 m/s²
//! assert!((v - 25.0).abs() < 1e-10);  // v = 25.0 m/s
//! ```
//!
//! ### How It Works
//!
//! 1. **Analysis Phase**: Build dependency graph showing which variables appear in which equations
//! 2. **Planning Phase**: Determine solving order using topological sort to avoid circular dependencies
//! 3. **Execution Phase**: Solve equations sequentially, substituting known values from earlier steps
//! 4. **Verification Phase**: Validate that all solutions satisfy the original equations
//!
//! ### Chained Solving
//!
//! The solver automatically chains solutions when one equation's result is needed by another:
//!
//! ```rust,ignore
//! use thales::{EquationSystem, SystemContext, MultiEquationSolver, parse_equation};
//!
//! let mut system = EquationSystem::new();
//! system.add_equation("step1", parse_equation("a = b + 2").unwrap());
//! system.add_equation("step2", parse_equation("c = a * 3").unwrap());
//! system.add_equation("step3", parse_equation("d = c - 1").unwrap());
//!
//! let context = SystemContext::new()
//!     .with_known_value("b", 5.0)
//!     .with_target("d");
//!
//! let solver = MultiEquationSolver::new();
//! let solution = solver.solve(&system, &context).unwrap();
//!
//! // Automatically solves: a = 7, then c = 21, then d = 20
//! assert_eq!(solution.get_numeric("d").unwrap(), 20.0);
//! ```
//!
//! ## Solver Selection Strategy
//!
//! [`SmartSolver`](crate::SmartSolver) selects the appropriate method based on equation structure:
//!
//! | Equation Pattern | Solver Used | Example |
//! |------------------|-------------|---------|
//! | Linear: ax + b = c | Algebraic manipulation | 3x + 5 = 14 |
//! | Quadratic: ax² + bx + c = 0 | Discriminant formula | x² - 5x + 6 = 0 |
//! | Polynomial: Σ(aₙxⁿ) = 0 | Companion matrix | x³ - 6x² + 11x - 6 = 0 |
//! | Transcendental | Symbolic inverse or numerical | sin(x) = 0.5 |
//! | Implicit/Complex | Numerical root-finding | x = cos(x) |
//!
//! ## Common Patterns and Tips
//!
//! ### Pattern 1: Parse-Solve-Extract
//!
//! Standard workflow for string-based equations:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//!
//! fn solve_equation_string(eq_str: &str, var: &str) -> Option<f64> {
//!     let equation = parse_equation(eq_str).ok()?;
//!     let solver = SmartSolver::new();
//!     let (solution, _) = solver.solve(&equation, &var.into()).ok()?;
//!
//!     match solution {
//!         Solution::Unique(expr) => expr.evaluate(&Default::default()),
//!         _ => None,
//!     }
//! }
//!
//! assert_eq!(solve_equation_string("2*x + 3 = 11", "x"), Some(4.0));
//! ```
//!
//! ### Pattern 2: With Known Values
//!
//! Solve parametric equations by providing values:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver};
//! use std::collections::HashMap;
//!
//! let eq = parse_equation("a*x + b = c").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! // Solution is parametric: x = (c - b) / a
//! let mut known = HashMap::new();
//! known.insert("a".to_string(), 2.0);
//! known.insert("b".to_string(), 3.0);
//! known.insert("c".to_string(), 11.0);
//!
//! // Evaluate with known values
//! if let Solution::Unique(expr) = solution {
//!     let result = expr.evaluate(&known).unwrap();
//!     assert_eq!(result, 4.0);
//! }
//! ```
//!
//! ### Pattern 3: Solution Verification
//!
//! Always verify critical solutions by substitution:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, Solution};
//! use std::collections::HashMap;
//!
//! let eq = parse_equation("x^2 - 5*x + 6 = 0").unwrap();
//! let solver = SmartSolver::new();
//! let (solution, _) = solver.solve(&eq, &"x".into()).unwrap();
//!
//! if let Solution::Multiple(roots) = solution {
//!     for root in roots {
//!         // Substitute back into original equation
//!         let mut values = HashMap::new();
//!         let root_value = root.evaluate(&values).unwrap();
//!         values.insert("x".to_string(), root_value);
//!
//!         let lhs = eq.left.evaluate(&values).unwrap();
//!         let rhs = eq.right.evaluate(&values).unwrap();
//!         assert!((lhs - rhs).abs() < 1e-10, "Solution verification failed");
//!     }
//! }
//! ```
//!
//! ### Pattern 4: Error Handling
//!
//! Handle cases where equations can't be solved:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartSolver, Solver, SolverError};
//!
//! fn try_solve(eq_str: &str, var: &str) -> Result<f64, SolverError> {
//!     let equation = parse_equation(eq_str)
//!         .map_err(|_| SolverError::Other("Parse error".into()))?;
//!
//!     let solver = SmartSolver::new();
//!     let (solution, _) = solver.solve(&equation, &var.into())?;
//!
//!     match solution {
//!         Solution::Unique(expr) => {
//!             expr.evaluate(&Default::default())
//!                 .ok_or_else(|| SolverError::Other("Evaluation failed".into()))
//!         }
//!         Solution::None => Err(SolverError::NoSolution),
//!         Solution::Infinite => Err(SolverError::InfiniteSolutions),
//!         _ => Err(SolverError::Other("Complex solution type".into())),
//!     }
//! }
//! ```
//!
//! ## Performance Considerations
//!
//! - Linear equations: O(1) - constant time algebraic manipulation
//! - Quadratic equations: O(1) - discriminant calculation
//! - Polynomial degree d: O(d²) - companion matrix eigenvalue computation
//! - Numerical methods: O(k) - k iterations until convergence
//!
//! For performance-critical applications:
//! 1. Reuse solver instances (they're stateless)
//! 2. Cache parsed equations when solving repeatedly
//! 3. Use direct AST construction instead of parsing strings
//! 4. Provide initial guesses for numerical solvers
//!
//! ## See Also
//!
//! - [`crate::guides::numerical_methods`] - Numerical root-finding when symbolic solving fails
//! - [`crate::guides::calculus_operations`] - Working with derivatives and integrals
//! - [`crate::guides::error_handling`] - Handling `ThalesError` and `SolverError`
//! - [`crate::SmartSolver`] - Automatic solver selection
//! - [`crate::MultiEquationSolver`] - Multi-equation system solving
//! - [`crate::Solution`] - Solution type documentation
//! - [`crate::Variable`] - Variable representation
