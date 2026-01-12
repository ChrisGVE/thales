//! # Numerical Methods Guide
//!
//! This guide covers numerical root-finding methods for equations that cannot be solved
//! symbolically. When algebraic manipulation fails, numerical methods provide approximate
//! solutions through iterative refinement.
//!
//! ## When to Use Numerical Methods
//!
//! **Use numerical methods when:**
//! - Transcendental equations mix different function types: `e^x = x + 2`
//! - Implicit equations cannot be rearranged: `x = cos(x)`
//! - High-degree polynomials lack closed-form solutions: `x^7 - 3x^5 + 2x - 1 = 0`
//! - Symbolic solvers fail or produce intractable expressions
//!
//! **Use symbolic methods (see [`crate::guides::solving_equations`]) when:**
//! - Linear equations: `3x + 5 = 14`
//! - Quadratic equations: `x^2 - 5x + 6 = 0`
//! - Simple polynomial equations: `x^3 - 8 = 0`
//! - Equations with symbolic inverses: `sin(x) = 0.5` → `x = arcsin(0.5)`
//!
//! ## Quick Start: Smart Numerical Solver
//!
//! The [`SmartNumericalSolver`](crate::SmartNumericalSolver) automatically selects
//! the best method for your equation:
//!
//! ```rust,ignore
//! use thales::{parse_equation, SmartNumericalSolver, NumericalConfig};
//!
//! // Solve: e^x = x + 2
//! let equation = parse_equation("exp(x) = x + 2").unwrap();
//! let config = NumericalConfig::default();
//! let solver = SmartNumericalSolver::new(config);
//!
//! // Initial guess near the expected root
//! let solution = solver.solve_for(&equation, "x", 1.0).unwrap();
//!
//! println!("x ≈ {}", solution.value);
//! println!("Converged: {}", solution.converged);
//! println!("Iterations: {}", solution.iterations);
//! println!("Residual: {:.2e}", solution.residual);
//! ```
//!
//! The smart solver analyzes the equation and chooses between Newton-Raphson,
//! bisection, secant, or Brent's method based on characteristics like smoothness
//! and derivative availability.
//!
//! ## Newton-Raphson Method
//!
//! **Best for:** Smooth functions with good initial guesses
//!
//! The Newton-Raphson method uses the tangent line approximation for rapid convergence:
//!
//! **x_{n+1} = x_n - f(x_n) / f'(x_n)**
//!
//! ### Key Advantages
//!
//! - **Quadratic convergence**: Doubles accuracy each iteration near the root
//! - **Exact derivatives**: Uses symbolic differentiation, not finite differences
//! - **Fast**: Typically converges in 3-7 iterations
//! - **Precise**: Can achieve machine precision (1e-15 tolerance)
//!
//! ### Example: Finding Square Roots
//!
//! ```rust,ignore
//! use thales::numerical::{NewtonRaphson, NumericalConfig};
//! use thales::ast::{Equation, Expression, Variable};
//!
//! // Solve x^2 = 5 to find √5
//! let equation = Equation::new(
//!     "sqrt5",
//!     Expression::Power(
//!         Box::new(Expression::Variable(Variable::new("x"))),
//!         Box::new(Expression::Integer(2))
//!     ),
//!     Expression::Integer(5)
//! );
//!
//! let solver = NewtonRaphson::with_default_config();
//! let (solution, _path) = solver.solve(&equation, &Variable::new("x")).unwrap();
//!
//! assert!((solution.value - 2.236067977).abs() < 1e-6);
//! assert!(solution.converged);
//! ```
//!
//! ### When Newton-Raphson Fails
//!
//! - **Zero derivative**: When f'(x) = 0 at iteration point (e.g., inflection points)
//! - **Poor initial guess**: Starting too far from the root may cause divergence
//! - **Multiple roots**: May jump between roots if they're close together
//! - **Discontinuities**: Method assumes smoothness
//!
//! ## Bisection Method
//!
//! **Best for:** Guaranteed convergence when root is bracketed
//!
//! The bisection method repeatedly halves the search interval, guaranteeing convergence
//! for continuous functions:
//!
//! ### Key Advantages
//!
//! - **Always converges**: If you bracket a root, bisection will find it
//! - **No derivatives**: Works for non-smooth functions
//! - **Robust**: Immune to poor initial guesses (within bracket)
//! - **Predictable**: Fixed number of iterations for given tolerance
//!
//! ### Example: Bracketing a Root
//!
//! ```rust,ignore
//! use thales::numerical::{BisectionMethod, NumericalConfig};
//! use thales::parse_equation;
//!
//! // Solve: x^3 - x - 2 = 0
//! let equation = parse_equation("x^3 - x - 2 = 0").unwrap();
//! let config = NumericalConfig::default();
//! let solver = BisectionMethod::new(config);
//!
//! // Bracket the root: f(0) < 0, f(2) > 0
//! let (solution, path) = solver.solve_bracketed(&equation, "x", 0.0, 2.0).unwrap();
//!
//! println!("Root: {}", solution.value);
//! println!("Iterations: {}", solution.iterations);
//! ```
//!
//! ### Convergence Rate
//!
//! Bisection converges linearly: each iteration reduces error by factor of 2.
//! To achieve tolerance ε from initial bracket [a, b]:
//!
//! **Iterations = log₂((b - a) / ε)**
//!
//! For tolerance 1e-10 and initial bracket width 2.0: ~33 iterations.
//!
//! ## Brent's Method
//!
//! **Best for:** Combining speed and reliability
//!
//! Brent's method is a hybrid that combines bisection's reliability with
//! faster interpolation methods (secant and inverse quadratic):
//!
//! ### Key Advantages
//!
//! - **Fast convergence**: Nearly as fast as Newton-Raphson
//! - **Guaranteed convergence**: Falls back to bisection when needed
//! - **No derivatives**: Uses only function values
//! - **Industry standard**: Used in MATLAB, SciPy, NumPy
//!
//! ### Example: Transcendental Equation
//!
//! ```rust,ignore
//! use thales::numerical::{BrentsMethod, NumericalConfig};
//! use thales::parse_equation;
//!
//! // Solve: cos(x) = x (implicit equation)
//! let equation = parse_equation("cos(x) = x").unwrap();
//! let config = NumericalConfig::default();
//! let solver = BrentsMethod::new(config);
//!
//! // Bracket: f(0) = cos(0) - 0 = 1 > 0, f(π/2) < 0
//! let (solution, _) = solver.solve_bracketed(&equation, "x", 0.0, 1.57).unwrap();
//!
//! // Solution: x ≈ 0.739085 (Dottie number)
//! assert!((solution.value - 0.739085).abs() < 1e-5);
//! ```
//!
//! Brent's method typically converges in 5-10 iterations, faster than bisection
//! but more robust than Newton-Raphson.
//!
//! ## Secant Method
//!
//! **Best for:** When derivatives are unavailable or expensive
//!
//! The secant method approximates the derivative using two previous points:
//!
//! **x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))**
//!
//! ### Key Advantages
//!
//! - **No derivatives**: Uses finite difference approximation
//! - **Superlinear convergence**: Faster than bisection, nearly quadratic
//! - **Two initial points**: More flexible than Newton-Raphson's single guess
//!
//! ### Example: Using Secant Method
//!
//! ```rust,ignore
//! use thales::numerical::{SecantMethod, NumericalConfig};
//! use thales::parse_equation;
//!
//! // Solve: e^x = 3x
//! let equation = parse_equation("exp(x) = 3*x").unwrap();
//! let config = NumericalConfig::default();
//! let solver = SecantMethod::new(config);
//!
//! // Provide two initial guesses
//! let (solution, _) = solver.solve_with_two_points(&equation, "x", 0.5, 1.0).unwrap();
//!
//! println!("Root: {}", solution.value);
//! ```
//!
//! Convergence rate is approximately 1.618 (golden ratio), between linear and quadratic.
//!
//! ## SmartNumericalSolver: Automatic Method Selection
//!
//! The [`SmartNumericalSolver`](crate::SmartNumericalSolver) analyzes your equation
//! and automatically chooses the best method:
//!
//! ### Selection Strategy
//!
//! 1. **Check for symbolic derivatives**: If available and well-defined → Newton-Raphson
//! 2. **Check for continuity**: If well-behaved → Brent's method
//! 3. **Check for smoothness**: If smooth but complex → Secant method
//! 4. **Default fallback**: Bisection with automatic bracketing
//!
//! ### Example: Automatic Selection
//!
//! ```rust,ignore
//! use thales::{SmartNumericalSolver, NumericalConfig, parse_equation};
//!
//! let config = NumericalConfig::default();
//! let solver = SmartNumericalSolver::new(config);
//!
//! // Smart solver chooses the right method for each equation
//! let eq1 = parse_equation("x^2 = 5").unwrap();
//! let sol1 = solver.solve_for(&eq1, "x", 2.0).unwrap();
//! // → Uses Newton-Raphson (smooth, derivative exists)
//!
//! let eq2 = parse_equation("abs(x) = 2").unwrap();
//! let sol2 = solver.solve_for(&eq2, "x", 1.0).unwrap();
//! // → Uses bisection (non-smooth at x=0)
//!
//! let eq3 = parse_equation("x = cos(x)").unwrap();
//! let sol3 = solver.solve_for(&eq3, "x", 0.5).unwrap();
//! // → Uses Brent's method (transcendental, well-behaved)
//! ```
//!
//! ## Configuration Options
//!
//! All solvers use [`NumericalConfig`](crate::NumericalConfig) for control:
//!
//! ### Tolerance
//!
//! Controls when to stop iterating. The algorithm stops when either:
//! - **Residual criterion**: |f(x)| < tolerance
//! - **Step criterion**: |x_{n+1} - x_n| < tolerance
//!
//! ```rust,ignore
//! use thales::NumericalConfig;
//!
//! // High precision (machine epsilon limit)
//! let precise = NumericalConfig {
//!     tolerance: 1e-15,
//!     ..Default::default()
//! };
//!
//! // Standard precision (engineering default)
//! let standard = NumericalConfig {
//!     tolerance: 1e-10,
//!     ..Default::default()
//! };
//!
//! // Fast approximation (plotting, visualization)
//! let fast = NumericalConfig {
//!     tolerance: 1e-6,
//!     ..Default::default()
//! };
//! ```
//!
//! ### Maximum Iterations
//!
//! Prevents infinite loops for divergent or slowly converging cases:
//!
//! ```rust,ignore
//! use thales::NumericalConfig;
//!
//! // Conservative limit (fail fast)
//! let conservative = NumericalConfig {
//!     max_iterations: 100,
//!     ..Default::default()
//! };
//!
//! // Standard limit (default)
//! let standard = NumericalConfig {
//!     max_iterations: 1000,
//!     ..Default::default()
//! };
//!
//! // Generous limit (difficult problems)
//! let generous = NumericalConfig {
//!     max_iterations: 10000,
//!     ..Default::default()
//! };
//! ```
//!
//! ### Initial Guess
//!
//! Critical for Newton-Raphson and secant methods:
//!
//! ```rust,ignore
//! use thales::NumericalConfig;
//!
//! // Provide good initial guess
//! let with_guess = NumericalConfig {
//!     initial_guess: Some(2.0),  // Near expected root
//!     ..Default::default()
//! };
//!
//! // Let solver choose default (1.0)
//! let auto_guess = NumericalConfig {
//!     initial_guess: None,
//!     ..Default::default()
//! };
//! ```
//!
//! **Tips for choosing initial guesses:**
//! - Plot the function to visualize roots
//! - Use physical intuition or problem context
//! - Try multiple guesses to find different roots
//! - For polynomials, try values around sign changes
//!
//! ## Handling Convergence Failures
//!
//! Numerical methods can fail. Handle errors gracefully:
//!
//! ### Error Types
//!
//! The [`NumericalError`](crate::numerical::NumericalError) enum covers all failure modes:
//!
//! ```rust,ignore
//! use thales::{SmartNumericalSolver, NumericalConfig, numerical::NumericalError};
//!
//! let config = NumericalConfig::default();
//! let solver = SmartNumericalSolver::new(config);
//! let equation = parse_equation("x^2 + 1 = 0").unwrap();  // No real roots
//!
//! match solver.solve_for(&equation, "x", 0.0) {
//!     Ok(solution) => println!("Root: {}", solution.value),
//!     Err(NumericalError::NoConvergence) => {
//!         eprintln!("Failed to converge within iteration limit");
//!         eprintln!("Try: increase max_iterations or adjust initial_guess");
//!     }
//!     Err(NumericalError::Unstable) => {
//!         eprintln!("Numerical instability detected");
//!         eprintln!("Try: different method or better initial guess");
//!     }
//!     Err(NumericalError::InvalidInitialGuess) => {
//!         eprintln!("Initial guess is outside valid domain");
//!         eprintln!("Try: different starting point");
//!     }
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```
//!
//! ### Recovery Strategies
//!
//! When a method fails, try these recovery techniques:
//!
//! **Strategy 1: Try different initial guesses**
//!
//! ```rust,ignore
//! let guesses = vec![0.0, 1.0, -1.0, 5.0, -5.0];
//! for &guess in &guesses {
//!     if let Ok(solution) = solver.solve_for(&equation, "x", guess) {
//!         println!("Found root: {}", solution.value);
//!         break;
//!     }
//! }
//! ```
//!
//! **Strategy 2: Switch methods**
//!
//! ```rust,ignore
//! // Try Newton-Raphson first (fast)
//! let newton = NewtonRaphson::with_default_config();
//! if let Ok(solution) = newton.solve(&equation, &Variable::new("x")) {
//!     return solution;
//! }
//!
//! // Fall back to bisection (robust)
//! let bisection = BisectionMethod::new(config);
//! let solution = bisection.solve_bracketed(&equation, "x", -10.0, 10.0).unwrap();
//! ```
//!
//! **Strategy 3: Adjust tolerance and iterations**
//!
//! ```rust,ignore
//! let relaxed = NumericalConfig {
//!     max_iterations: 5000,
//!     tolerance: 1e-8,  // Slightly relaxed
//!     ..Default::default()
//! };
//! ```
//!
//! **Strategy 4: Check for real roots**
//!
//! ```rust,ignore
//! // Equation x^2 + 1 = 0 has no real roots
//! // Check discriminant or use symbolic solver first
//! if equation_has_real_roots(&equation) {
//!     let solution = solver.solve_for(&equation, "x", 0.0)?;
//! } else {
//!     println!("No real roots exist");
//! }
//! ```
//!
//! ## Method Selection Flowchart
//!
//! ```text
//! Need to solve f(x) = 0 numerically?
//!         |
//!         v
//! Do you know derivative f'(x)?
//!         |
//!    Yes  |  No
//!         |
//!    +----+----+
//!    |         |
//!    v         v
//! Newton-  Can bracket?
//! Raphson      |
//!         Yes  |  No
//!              |
//!         +----+----+
//!         |         |
//!         v         v
//!     Brent's   Secant
//!     Method    Method
//!
//! Unsure? → SmartNumericalSolver
//! ```
//!
//! ## Performance Comparison
//!
//! Typical convergence for solving x^2 = 5 (√5 ≈ 2.236):
//!
//! | Method | Iterations | Evaluations | Best For |
//! |--------|-----------|-------------|----------|
//! | Newton-Raphson | 4-5 | 8-10 (f + f') | Smooth functions |
//! | Secant | 6-8 | 6-8 (f only) | No derivatives |
//! | Brent | 7-9 | 7-9 (f only) | Robustness |
//! | Bisection | 30-33 | 30-33 (f only) | Guaranteed convergence |
//!
//! ## See Also
//!
//! - [`crate::guides::solving_equations`] - Symbolic equation solving
//! - [`crate::SmartNumericalSolver`] - Automatic method selection
//! - [`crate::NumericalSolution`] - Solution result type
//! - [`crate::NumericalConfig`] - Configuration options
//! - [`crate::numerical`] - Full numerical methods module
