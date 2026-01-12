//! # Series Expansions Guide
//!
//! This guide covers working with power series in thales, including Taylor series,
//! Maclaurin series, Laurent series for functions with singularities, and asymptotic
//! expansions for approximating function behavior.
//!
//! ## Overview
//!
//! The [`crate::series`] module provides comprehensive series expansion capabilities:
//!
//! - **Taylor series**: Expand functions around arbitrary center points
//! - **Maclaurin series**: Special case of Taylor series centered at x=0
//! - **Laurent series**: Handle functions with poles and singularities
//! - **Asymptotic expansions**: Approximate behavior as x→∞ or x→0
//! - **Big-O notation**: Track error terms and convergence
//!
//! ## Quick Start: Maclaurin Series
//!
//! The simplest expansion is a Maclaurin series (Taylor at x=0):
//!
//! ```rust,ignore
//! use thales::{maclaurin, Expression, Variable, Function};
//!
//! let x = Variable::new("x");
//! let expr = Expression::Function(
//!     Function::Exp,
//!     vec![Expression::Variable(x.clone())]
//! );
//!
//! // Expand e^x to 5th order at x=0
//! let series = maclaurin(&expr, &x, 5).unwrap();
//!
//! // Result: 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + O(x⁶)
//! println!("{}", series.to_latex());
//! ```
//!
//! ## Taylor Series Around a Point
//!
//! For expansions around arbitrary center points a, use [`crate::series::taylor`]:
//!
//! ```rust,ignore
//! use thales::{taylor, Expression, Variable, Function};
//!
//! let x = Variable::new("x");
//! let expr = Expression::Function(
//!     Function::Ln,
//!     vec![Expression::Variable(x.clone())]
//! );
//!
//! // Expand ln(x) around x=1 to 4th order
//! let center = Expression::Integer(1);
//! let series = taylor(&expr, &x, &center, 4).unwrap();
//!
//! // Result: (x-1) - (x-1)²/2 + (x-1)³/3 - (x-1)⁴/4 + O((x-1)⁵)
//! ```
//!
//! **Why use Taylor series?**
//! - Approximate transcendental functions with polynomials
//! - Understand function behavior near a point
//! - Compute derivatives symbolically
//! - Numerical approximation when exact form unknown
//!
//! ## Common Function Expansions
//!
//! The [`crate::series`] module provides built-in expansions for standard functions:
//!
//! ```rust,ignore
//! use thales::{exp_series, sin_series, cos_series, ln_1_plus_x_series, arctan_series};
//!
//! // Exponential: e^x = 1 + x + x²/2! + x³/3! + ...
//! let exp = exp_series(5).unwrap();
//!
//! // Sine: sin(x) = x - x³/3! + x⁵/5! - ...
//! let sine = sin_series(7).unwrap();
//!
//! // Cosine: cos(x) = 1 - x²/2! + x⁴/4! - ...
//! let cosine = cos_series(6).unwrap();
//!
//! // Natural log: ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
//! let log = ln_1_plus_x_series(4).unwrap();
//!
//! // Arctangent: arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
//! let arctan = arctan_series(9).unwrap();
//! ```
//!
//! ## Laurent Series for Functions with Poles
//!
//! When functions have singularities, Laurent series include negative powers:
//!
//! ```rust,ignore
//! use thales::{laurent, Expression, Variable, BinaryOp};
//!
//! let x = Variable::new("x");
//!
//! // Expand 1/(x-1) around x=1
//! let expr = Expression::Binary(
//!     BinaryOp::Div,
//!     Box::new(Expression::Integer(1)),
//!     Box::new(Expression::Binary(
//!         BinaryOp::Sub,
//!         Box::new(Expression::Variable(x.clone())),
//!         Box::new(Expression::Integer(1))
//!     ))
//! );
//!
//! let center = Expression::Integer(1);
//! let series = laurent(&expr, &x, &center, -1, 3).unwrap();
//!
//! // Result: 1/(x-1) + 0 + 0 + ... (simple pole at x=1)
//! ```
//!
//! **Key concepts:**
//! - **Principal part**: Terms with negative powers (singular part)
//! - **Regular part**: Terms with non-negative powers (analytic part)
//! - **Residue**: Coefficient of the -1 power term
//! - **Pole order**: Most negative power with non-zero coefficient
//!
//! ### Finding Residues
//!
//! The residue is critical for complex integration via residue theorem:
//!
//! ```rust,ignore
//! use thales::{residue, Expression, Variable, BinaryOp, Function};
//!
//! let x = Variable::new("x");
//! let expr = Expression::Binary(
//!     BinaryOp::Div,
//!     Box::new(Expression::Function(
//!         Function::Sin,
//!         vec![Expression::Variable(x.clone())]
//!     )),
//!     Box::new(Expression::Variable(x.clone()))
//! );
//!
//! let center = Expression::Integer(0);
//! let res = residue(&expr, &x, &center).unwrap();
//! // For sin(x)/x at x=0, residue is 0 (removable singularity)
//! ```
//!
//! ## Asymptotic Expansions
//!
//! For behavior as x→∞ or x→0⁺, use [`crate::series::asymptotic`]:
//!
//! ```rust,ignore
//! use thales::{asymptotic, AsymptoticDirection, Expression, Variable, BinaryOp, Function};
//!
//! let x = Variable::new("x");
//!
//! // Stirling's approximation: ln(n!) ~ n·ln(n) - n as n→∞
//! let expr = Expression::Function(
//!     Function::Ln,
//!     vec![Expression::Variable(x.clone())]
//! );
//!
//! let expansion = asymptotic(&expr, &x, AsymptoticDirection::Infinity, 3).unwrap();
//!
//! // Returns expansion with error estimate
//! println!("Error term: {}", expansion.error_term);
//! ```
//!
//! **Asymptotic directions:**
//! - [`AsymptoticDirection::Infinity`]: x→∞
//! - [`AsymptoticDirection::ZeroPlus`]: x→0⁺
//! - [`AsymptoticDirection::ZeroMinus`]: x→0⁻
//!
//! ## Working with Series Terms
//!
//! Access individual terms and coefficients:
//!
//! ```rust,ignore
//! use thales::{maclaurin, Expression, Variable, Function};
//!
//! let x = Variable::new("x");
//! let expr = Expression::Function(
//!     Function::Exp,
//!     vec![Expression::Variable(x.clone())]
//! );
//! let series = maclaurin(&expr, &x, 4).unwrap();
//!
//! // Get specific term
//! if let Some(term) = series.get_term(2) {
//!     println!("x² coefficient: {}", term.coefficient); // 1/2
//!     println!("power: {}", term.power); // 2
//! }
//!
//! // Count non-zero terms
//! println!("Number of terms: {}", series.term_count());
//!
//! // Convert to polynomial expression
//! let poly = series.to_expression();
//! ```
//!
//! ## Convergence and Truncation
//!
//! Series approximations have error bounds tracked via Big-O notation:
//!
//! ```rust,ignore
//! use thales::{maclaurin, Expression, Variable, Function, RemainderTerm};
//!
//! let x = Variable::new("x");
//! let expr = Expression::Function(
//!     Function::Sin,
//!     vec![Expression::Variable(x.clone())]
//! );
//!
//! let series = maclaurin(&expr, &x, 5).unwrap();
//!
//! // Check remainder term
//! if let Some(remainder) = &series.remainder {
//!     match remainder {
//!         RemainderTerm::BigO { order } => {
//!             println!("Error is O(x^{})", order);
//!         }
//!         RemainderTerm::Lagrange { bound, order } => {
//!             println!("Error ≤ {} for order {}", bound, order);
//!         }
//!     }
//! }
//! ```
//!
//! **Convergence guidelines:**
//! - Taylor/Maclaurin series converge within radius of convergence
//! - Higher orders give better accuracy near the center
//! - For e^x, sin(x), cos(x): infinite radius (converge everywhere)
//! - For ln(1+x): radius = 1 (converges for |x| < 1)
//! - For 1/(1-x): radius = 1 (geometric series)
//!
//! ## Related Modules
//!
//! - [`crate::series`]: Full series expansion API
//! - [`crate::limits`]: Compute limits using series
//! - [`crate::calculus_operations`]: Integration and differentiation
//! - [`crate::numerical`]: Numerical approximation methods
