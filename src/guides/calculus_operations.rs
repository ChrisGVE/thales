//! # Calculus Operations Guide
//!
//! This guide covers calculus capabilities in thales, from symbolic differentiation
//! to solving differential equations.
//!
//! ## Quick Reference
//!
//! | Operation | Function | Example |
//! |-----------|----------|---------|
//! | Differentiate | [`Expression::differentiate`] | `expr.differentiate("x")` |
//! | Indefinite integral | [`crate::integrate`] | `integrate(&expr, "x")` |
//! | Definite integral | [`crate::definite_integral`] | `definite_integral(&expr, "x", 0.0, 1.0)` |
//! | Limit | [`crate::limits::limit`] | `limit(&expr, "x", LimitPoint::Value(0.0))` |
//! | L'Hôpital's rule | [`crate::limits::limit_with_lhopital`] | `limit_with_lhopital(&expr, "x", point)` |
//! | Separable ODE | [`crate::ode::solve_separable`] | `solve_separable(&ode)` |
//! | Linear ODE | [`crate::ode::solve_linear`] | `solve_linear(&ode)` |
//! | Second-order ODE | [`crate::ode::solve_second_order_homogeneous`] | `solve_second_order_homogeneous(&ode)` |
//!
//! ## Overview
//!
//! The thales library provides comprehensive calculus support for symbolic computation:
//!
//! - **Differentiation**: Automatic symbolic differentiation using the chain rule
//! - **Integration**: Pattern-based symbolic integration with common techniques
//! - **Limits**: Direct substitution, L'Hôpital's rule, and limits at infinity
//! - **Differential Equations**: Separable, linear, and second-order ODEs
//!
//! All operations work on the [`Expression`] type from the AST module.
//!
//! # 1. Symbolic Differentiation
//!
//! The [`Expression::differentiate`] method computes derivatives symbolically using
//! standard calculus rules.
//!
//! ## Basic Differentiation
//!
//! ```rust,ignore
//! use thales::{Expression, Variable, BinaryOp, parse_expression};
//!
//! // d/dx(x^2) = 2x
//! let expr = parse_expression("x^2").unwrap();
//! let derivative = expr.differentiate("x");
//! // Result: 2*x
//!
//! // d/dx(sin(x)) = cos(x)
//! let expr = parse_expression("sin(x)").unwrap();
//! let derivative = expr.differentiate("x");
//! // Result: cos(x)
//! ```
//!
//! ## Chain Rule
//!
//! The differentiation engine automatically applies the chain rule:
//!
//! ```rust,ignore
//! use thales::parse_expression;
//!
//! // d/dx(sin(x^2)) = cos(x^2) * 2x
//! let expr = parse_expression("sin(x^2)").unwrap();
//! let derivative = expr.differentiate("x");
//! // Applies chain rule automatically
//! ```
//!
//! ## Product and Quotient Rules
//!
//! ```rust,ignore
//! use thales::parse_expression;
//!
//! // Product rule: d/dx(x * sin(x)) = sin(x) + x*cos(x)
//! let expr = parse_expression("x * sin(x)").unwrap();
//! let derivative = expr.differentiate("x");
//!
//! // Quotient rule: d/dx(sin(x)/x) = (x*cos(x) - sin(x))/x^2
//! let expr = parse_expression("sin(x)/x").unwrap();
//! let derivative = expr.differentiate("x");
//! ```
//!
//! ## Higher-Order Derivatives
//!
//! ```rust,ignore
//! use thales::parse_expression;
//!
//! // Second derivative: d²/dx²(x^3) = 6x
//! let expr = parse_expression("x^3").unwrap();
//! let first = expr.differentiate("x");
//! let second = first.differentiate("x");
//! // Result: 6*x
//! ```
//!
//! # 2. Indefinite Integration
//!
//! The [`crate::integrate`] function computes antiderivatives using pattern matching
//! and integration techniques.
//!
//! ## Basic Integration
//!
//! ```rust,ignore
//! use thales::{integrate, parse_expression};
//!
//! // ∫x^2 dx = x^3/3 + C
//! let expr = parse_expression("x^2").unwrap();
//! let integral = integrate(&expr, "x").unwrap();
//! // Result: x^3/3 (constant C not included)
//!
//! // ∫sin(x) dx = -cos(x) + C
//! let expr = parse_expression("sin(x)").unwrap();
//! let integral = integrate(&expr, "x").unwrap();
//! ```
//!
//! ## Sum and Difference Rules
//!
//! ```rust,ignore
//! use thales::{integrate, parse_expression};
//!
//! // ∫(x^2 + 3x + 1) dx = x^3/3 + 3x^2/2 + x + C
//! let expr = parse_expression("x^2 + 3*x + 1").unwrap();
//! let integral = integrate(&expr, "x").unwrap();
//! ```
//!
//! ## Exponential and Logarithmic
//!
//! ```rust,ignore
//! use thales::{integrate, parse_expression};
//!
//! // ∫e^x dx = e^x + C
//! let expr = parse_expression("exp(x)").unwrap();
//! let integral = integrate(&expr, "x").unwrap();
//!
//! // ∫1/x dx = ln|x| + C
//! let expr = parse_expression("1/x").unwrap();
//! let integral = integrate(&expr, "x").unwrap();
//! ```
//!
//! # 3. Definite Integrals
//!
//! The [`crate::definite_integral`] function evaluates integrals over a specific interval.
//!
//! ## Numeric Bounds
//!
//! ```rust,ignore
//! use thales::{definite_integral, parse_expression};
//!
//! // ∫₀¹ x^2 dx = 1/3
//! let expr = parse_expression("x^2").unwrap();
//! let result = definite_integral(&expr, "x", 0.0, 1.0).unwrap();
//! // Returns approximately 0.333...
//! ```
//!
//! ## With Step-by-Step Output
//!
//! ```rust,ignore
//! use thales::{definite_integral_with_steps, parse_expression};
//!
//! let expr = parse_expression("x^2").unwrap();
//! let (result, steps) = definite_integral_with_steps(&expr, "x", 0.0, 1.0).unwrap();
//! // steps contains the antiderivative and evaluation process
//! ```
//!
//! ## Improper Integrals
//!
//! ```rust,ignore
//! use thales::{improper_integral_to_infinity, parse_expression};
//!
//! // ∫₁^∞ 1/x^2 dx = 1
//! let expr = parse_expression("1/x^2").unwrap();
//! let result = improper_integral_to_infinity(&expr, "x", 1.0, true).unwrap();
//! ```
//!
//! # 4. Integration Techniques
//!
//! Advanced integration methods for complex integrands.
//!
//! ## Integration by Parts
//!
//! For integrals of the form ∫u dv = uv - ∫v du:
//!
//! ```rust,ignore
//! use thales::{integrate_by_parts, parse_expression};
//!
//! // ∫x·e^x dx
//! let u = parse_expression("x").unwrap();
//! let dv = parse_expression("exp(x)").unwrap();
//! let result = integrate_by_parts(&u, &dv, "x").unwrap();
//! // Result: x·e^x - e^x
//! ```
//!
//! With step-by-step explanation:
//!
//! ```rust,ignore
//! use thales::integrate_by_parts_with_steps;
//!
//! let u = parse_expression("x").unwrap();
//! let dv = parse_expression("exp(x)").unwrap();
//! let (result, steps) = integrate_by_parts_with_steps(&u, &dv, "x").unwrap();
//! // steps shows u, dv, du, v, and final computation
//! ```
//!
//! ## Tabular Integration
//!
//! For repeated integration by parts (useful for polynomial × exponential):
//!
//! ```rust,ignore
//! use thales::tabular_integration;
//!
//! // ∫x^3·e^x dx using tabular method
//! let u = parse_expression("x^3").unwrap();
//! let dv = parse_expression("exp(x)").unwrap();
//! let result = tabular_integration(&u, &dv, "x").unwrap();
//! ```
//!
//! ## Substitution Method
//!
//! For integrals requiring u-substitution:
//!
//! ```rust,ignore
//! use thales::{integrate_by_substitution, integrate_with_substitution, parse_expression};
//!
//! // Basic substitution with automatic pattern detection
//! let expr = parse_expression("2*x*exp(x^2)").unwrap();
//! let result = integrate_by_substitution(&expr, "x").unwrap();
//!
//! // Manual substitution: specify u = g(x)
//! let expr = parse_expression("sin(x)*cos(x)").unwrap();
//! let u = parse_expression("sin(x)").unwrap();
//! let result = integrate_with_substitution(&expr, &u, "x").unwrap();
//! ```
//!
//! # 5. Limits and L'Hôpital's Rule
//!
//! Compute limits using direct substitution, special forms, or L'Hôpital's rule.
//!
//! ## Basic Limits
//!
//! ```rust,ignore
//! use thales::limits::{limit, LimitPoint, LimitResult};
//! use thales::parse_expression;
//!
//! // lim_{x→2} x^2 = 4
//! let expr = parse_expression("x^2").unwrap();
//! let result = limit(&expr, "x", LimitPoint::Value(2.0)).unwrap();
//! if let LimitResult::Value(v) = result {
//!     assert!((v - 4.0).abs() < 1e-10);
//! }
//! ```
//!
//! ## Limits at Infinity
//!
//! ```rust,ignore
//! use thales::limits::{limit, LimitPoint};
//!
//! // lim_{x→∞} 1/x = 0
//! let expr = parse_expression("1/x").unwrap();
//! let result = limit(&expr, "x", LimitPoint::PositiveInfinity).unwrap();
//!
//! // lim_{x→∞} x^2 = ∞
//! let expr = parse_expression("x^2").unwrap();
//! let result = limit(&expr, "x", LimitPoint::PositiveInfinity).unwrap();
//! ```
//!
//! ## One-Sided Limits
//!
//! ```rust,ignore
//! use thales::limits::{limit_left, limit_right};
//!
//! // lim_{x→0⁺} 1/x = +∞
//! let expr = parse_expression("1/x").unwrap();
//! let right_limit = limit_right(&expr, "x", 0.0).unwrap();
//!
//! // lim_{x→0⁻} 1/x = -∞
//! let left_limit = limit_left(&expr, "x", 0.0).unwrap();
//! ```
//!
//! ## L'Hôpital's Rule
//!
//! For indeterminate forms (0/0, ∞/∞):
//!
//! ```rust,ignore
//! use thales::limits::{limit_with_lhopital, LimitPoint, LimitResult};
//!
//! // lim_{x→0} sin(x)/x = 1
//! // Direct substitution gives 0/0, so apply L'Hôpital: lim cos(x)/1 = 1
//! let expr = parse_expression("sin(x)/x").unwrap();
//! let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
//!
//! // lim_{x→0} (e^x - 1)/x = 1
//! let expr = parse_expression("(exp(x) - 1)/x").unwrap();
//! let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
//!
//! // lim_{x→0} (1 - cos(x))/x^2 = 1/2 (may need multiple applications)
//! let expr = parse_expression("(1 - cos(x))/x^2").unwrap();
//! let result = limit_with_lhopital(&expr, "x", LimitPoint::Value(0.0)).unwrap();
//! ```
//!
//! # 6. First-Order ODEs
//!
//! Solve first-order ordinary differential equations using separable or linear methods.
//!
//! ## Separable ODEs
//!
//! For equations of the form dy/dx = g(x)·h(y):
//!
//! ```rust,ignore
//! use thales::ode::{FirstOrderODE, solve_separable};
//! use thales::{Expression, Variable, BinaryOp};
//!
//! // dy/dx = x·y
//! // Separating: (1/y) dy = x dx
//! // Solution: ln|y| = x^2/2 + C
//! let x = Expression::Variable(Variable::new("x"));
//! let y = Expression::Variable(Variable::new("y"));
//! let rhs = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(y));
//! let ode = FirstOrderODE::new("y", "x", rhs);
//!
//! let solution = solve_separable(&ode).unwrap();
//! // solution.general_solution contains the result
//! // solution.steps contains step-by-step explanation
//! ```
//!
//! ## Linear First-Order ODEs
//!
//! For equations of the form dy/dx + P(x)·y = Q(x):
//!
//! ```rust,ignore
//! use thales::ode::{FirstOrderODE, solve_linear};
//! use thales::parse_expression;
//!
//! // dy/dx + y = x
//! // Rewrite as: dy/dx = -y + x
//! let rhs = parse_expression("-y + x").unwrap();
//! let ode = FirstOrderODE::new("y", "x", rhs);
//!
//! let solution = solve_linear(&ode).unwrap();
//! // Uses integrating factor μ(x) = e^(∫P dx)
//! // solution.method == "Integrating factor"
//! ```
//!
//! ## Initial Value Problems
//!
//! Solve ODEs with an initial condition y(x₀) = y₀:
//!
//! ```rust,ignore
//! use thales::ode::{FirstOrderODE, solve_ivp};
//! use thales::{Expression, parse_expression};
//!
//! // dy/dx = y, y(0) = 1
//! // General solution: y = C·e^x
//! // Particular solution: y = e^x
//! let rhs = parse_expression("y").unwrap();
//! let ode = FirstOrderODE::new("y", "x", rhs);
//!
//! let x0 = Expression::Integer(0);
//! let y0 = Expression::Integer(1);
//! let solution = solve_ivp(&ode, &x0, &y0).unwrap();
//! // Determines C from initial condition
//! ```
//!
//! # 7. Second-Order ODEs
//!
//! Solve second-order linear ODEs with constant coefficients: a·y'' + b·y' + c·y = f(x).
//!
//! ## Homogeneous Case
//!
//! When f(x) = 0, solve using the characteristic equation:
//!
//! ```rust,ignore
//! use thales::ode::{SecondOrderODE, solve_second_order_homogeneous};
//!
//! // y'' + y = 0
//! // Characteristic equation: r^2 + 1 = 0 → r = ±i
//! // Solution: y = C1·cos(x) + C2·sin(x)
//! let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
//! let solution = solve_second_order_homogeneous(&ode).unwrap();
//!
//! // solution.roots contains the characteristic roots
//! // solution.homogeneous_solution is the general solution
//! ```
//!
//! ## Types of Solutions
//!
//! The solution form depends on the characteristic roots:
//!
//! ### Distinct Real Roots (Δ > 0)
//!
//! ```rust,ignore
//! use thales::ode::{SecondOrderODE, solve_second_order_homogeneous, RootType};
//!
//! // y'' - y = 0
//! // Characteristic: r^2 - 1 = 0 → r = ±1
//! // Solution: y = C1·e^x + C2·e^(-x)
//! let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, -1.0);
//! let solution = solve_second_order_homogeneous(&ode).unwrap();
//! assert_eq!(solution.roots.root_type, RootType::TwoDistinctReal);
//! ```
//!
//! ### Repeated Root (Δ = 0)
//!
//! ```rust,ignore
//! use thales::ode::{SecondOrderODE, solve_second_order_homogeneous, RootType};
//!
//! // y'' - 2y' + y = 0
//! // Characteristic: r^2 - 2r + 1 = 0 → r = 1 (double)
//! // Solution: y = (C1 + C2·x)·e^x
//! let ode = SecondOrderODE::homogeneous("y", "x", 1.0, -2.0, 1.0);
//! let solution = solve_second_order_homogeneous(&ode).unwrap();
//! assert_eq!(solution.roots.root_type, RootType::RepeatedReal);
//! ```
//!
//! ### Complex Conjugate Roots (Δ < 0)
//!
//! ```rust,ignore
//! use thales::ode::{SecondOrderODE, solve_second_order_homogeneous, RootType};
//!
//! // y'' + 4y' + 5y = 0
//! // Characteristic: r^2 + 4r + 5 = 0 → r = -2 ± i
//! // Solution: y = e^(-2x)·(C1·cos(x) + C2·sin(x))
//! let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 4.0, 5.0);
//! let solution = solve_second_order_homogeneous(&ode).unwrap();
//! assert_eq!(solution.roots.root_type, RootType::ComplexConjugate);
//! ```
//!
//! ## Second-Order IVPs
//!
//! Solve with initial conditions y(x₀) = y₀, y'(x₀) = y'₀:
//!
//! ```rust,ignore
//! use thales::ode::{SecondOrderODE, solve_second_order_ivp};
//!
//! // y'' + y = 0, y(0) = 1, y'(0) = 0
//! // General: y = C1·cos(x) + C2·sin(x)
//! // From conditions: C1 = 1, C2 = 0
//! // Particular: y = cos(x)
//! let ode = SecondOrderODE::homogeneous("y", "x", 1.0, 0.0, 1.0);
//! let solution = solve_second_order_ivp(&ode, 0.0, 1.0, 0.0).unwrap();
//! ```
//!
//! # Best Practices
//!
//! ## Simplification
//!
//! Always simplify expressions after calculus operations:
//!
//! ```rust,ignore
//! use thales::parse_expression;
//!
//! let expr = parse_expression("x^2").unwrap();
//! let derivative = expr.differentiate("x").simplify();
//! // Cleaner result: 2*x instead of complex AST
//! ```
//!
//! ## Error Handling
//!
//! Handle integration failures gracefully:
//!
//! ```rust,ignore
//! use thales::{integrate, IntegrationError};
//! use thales::parse_expression;
//!
//! let expr = parse_expression("abs(x)").unwrap();
//! match integrate(&expr, "x") {
//!     Ok(result) => println!("Integral: {}", result),
//!     Err(IntegrationError::CannotIntegrate(msg)) => {
//!         println!("Cannot integrate symbolically: {}", msg);
//!         // Fall back to numerical integration
//!     }
//!     Err(e) => println!("Error: {}", e),
//! }
//! ```
//!
//! ## Numerical Fallback
//!
//! Use numerical methods when symbolic integration fails:
//!
//! ```rust,ignore
//! use thales::{definite_integral_with_fallback, parse_expression};
//!
//! let expr = parse_expression("exp(-x^2)").unwrap(); // Gaussian - no closed form
//! let result = definite_integral_with_fallback(&expr, "x", 0.0, 1.0).unwrap();
//! // Automatically falls back to numerical integration
//! ```
//!
//! # See Also
//!
//! - [`Expression`]: The AST type used for all expressions
//! - [`crate::integration`]: Integration module with all techniques
//! - [`crate::limits`]: Limit evaluation and L'Hôpital's rule
//! - [`crate::ode`]: ODE solvers (first and second order)
//! - [`crate::numerical`]: Numerical methods for root finding
//! - [`crate::series`]: Taylor/Maclaurin series expansions
//!
//! [`Expression`]: crate::ast::Expression
//! [`Expression::differentiate`]: crate::ast::Expression::differentiate
