//! # Error Handling in Thales
//!
//! Comprehensive guide to working with errors in the thales library,
//! covering the unified [`crate::ThalesError`] type, module-specific
//! error types, conversions, pattern matching, and recovery strategies.
//!
//! ## Table of Contents
//!
//! 1. [Overview](#overview-of-thaleserror)
//! 2. [Module-Specific Errors](#module-specific-error-types)
//! 3. [Error Conversions](#converting-between-error-types)
//! 4. [Pattern Matching](#pattern-matching-on-errors)
//! 5. [The ? Operator](#the--operator-with-thales-errors)
//! 6. [Error Recovery](#error-recovery-strategies)
//! 7. [Logging and Debugging](#logging-and-debugging-errors)
//!
//! ---
//!
//! ## Overview of ThalesError
//!
//! The [`crate::ThalesError`] enum is the unified error type for the entire
//! thales library. It wraps all module-specific error types, providing a
//! consistent interface for error handling across the library.
//!
//! ```rust
//! use thales::ThalesError;
//!
//! // All thales operations can return ThalesError
//! fn process_math() -> Result<(), ThalesError> {
//!     // Future API will use ThalesError consistently
//!     Ok(())
//! }
//! ```
//!
//! **Key characteristics:**
//! - **Non-exhaustive**: New error variants may be added in future versions
//! - **Display + Debug**: All variants implement standard error traits
//! - **Error source**: Provides access to underlying error via `source()`
//! - **Automatic conversions**: Module errors convert to `ThalesError` via `From`
//!
//! ---
//!
//! ## Module-Specific Error Types
//!
//! Each module defines its own error type with variants specific to that domain.
//! Understanding these types helps you handle errors precisely.
//!
//! ### Parser Errors: [`crate::parser::ParseError`]
//!
//! Errors from parsing expressions and equations from strings.
//!
//! ```rust
//! use thales::{parse_expression, parser::ParseError};
//!
//! let result = parse_expression("2 + + 3");
//! match result {
//!     Err(errors) => {
//!         for err in errors {
//!             match err {
//!                 ParseError::UnexpectedCharacter { pos, found } => {
//!                     println!("Unexpected '{}' at position {}", found, pos);
//!                 }
//!                 ParseError::UnexpectedEndOfInput { pos, expected } => {
//!                     println!("Expected {} at position {}", expected, pos);
//!                 }
//!                 _ => println!("Other parse error: {}", err),
//!             }
//!         }
//!     }
//!     Ok(expr) => println!("Parsed: {:?}", expr),
//! }
//! ```
//!
//! ### Solver Errors: [`crate::solver::SolverError`]
//!
//! Errors from symbolic equation solving.
//!
//! ```rust
//! use thales::{solver::SolverError, Equation, Expression};
//!
//! fn handle_solver_error(err: SolverError) {
//!     match err {
//!         SolverError::NoSolution => {
//!             println!("Equation is inconsistent (e.g., 0 = 5)");
//!         }
//!         SolverError::InfiniteSolutions => {
//!             println!("Equation is an identity (e.g., x = x)");
//!         }
//!         SolverError::CannotSolve(msg) => {
//!             println!("Cannot solve: {}", msg);
//!         }
//!         SolverError::DivisionByZero => {
//!             println!("Division by zero during solving");
//!         }
//!         _ => println!("Other solver error: {:?}", err),
//!     }
//! }
//! ```
//!
//! ### Series Errors: [`crate::series::SeriesError`]
//!
//! Errors from Taylor/Maclaurin series expansions.
//!
//! ```rust
//! use thales::{series::SeriesError, taylor};
//!
//! fn handle_series_error(err: SeriesError) {
//!     match err {
//!         SeriesError::CannotExpand(msg) => {
//!             println!("Cannot expand: {}", msg);
//!         }
//!         SeriesError::InvalidCenter(msg) => {
//!             println!("Invalid center: {}", msg);
//!         }
//!         SeriesError::DivisionByZero => {
//!             println!("Division by zero in expansion");
//!         }
//!         SeriesError::DerivativeFailed(msg) => {
//!             println!("Differentiation failed: {}", msg);
//!         }
//!         _ => println!("Other series error: {}", err),
//!     }
//! }
//! ```
//!
//! ### Other Module Errors
//!
//! - [`crate::numerical::NumericalError`]: Convergence failures, invalid bounds
//! - [`crate::integration::IntegrationError`]: Integration failures, domain errors
//! - [`crate::matrix::MatrixError`]: Dimension mismatches, singular matrices
//! - [`crate::ode::ODEError`]: ODE solving failures, invalid initial conditions
//! - [`crate::inequality::InequalityError`]: Invalid inequalities, unbounded solutions
//! - [`crate::limits::LimitError`]: Limit evaluation failures, indeterminate forms
//! - [`crate::precision::EvalError`]: Evaluation errors, precision issues
//! - [`crate::special::SpecialFunctionError`]: Special function domain errors
//! - [`crate::partial_fractions::DecomposeError`]: Decomposition failures
//! - [`crate::latex::LaTeXParseError`]: LaTeX parsing errors
//! - [`crate::equation_system::SystemError`]: System solving failures
//! - [`crate::equation_system::NonlinearSystemSolverError`]: Nonlinear system errors
//!
//! ---
//!
//! ## Converting Between Error Types
//!
//! All module-specific errors automatically convert to [`crate::ThalesError`]
//! via the `From` trait. This enables seamless error propagation.
//!
//! ```rust
//! use thales::{ThalesError, parse_expression, parser::ParseError};
//!
//! // Module error automatically converts to ThalesError
//! fn parse_and_process(input: &str) -> Result<(), ThalesError> {
//!     // parse_expression returns Result<_, Vec<ParseError>>
//!     // We'll handle the first error for demonstration
//!     let expr = parse_expression(input)
//!         .map_err(|errors| {
//!             // Convert first ParseError to ThalesError
//!             ThalesError::from(errors.into_iter().next().unwrap())
//!         })?;
//!
//!     // Process expr...
//!     Ok(())
//! }
//! ```
//!
//! **Manual conversions:**
//!
//! ```rust
//! use thales::{ThalesError, solver::SolverError, series::SeriesError};
//!
//! // Explicit conversion using From/Into
//! let solver_err = SolverError::NoSolution;
//! let thales_err: ThalesError = solver_err.into();
//!
//! // Explicit conversion using From
//! let series_err = SeriesError::DivisionByZero;
//! let thales_err = ThalesError::from(series_err);
//! ```
//!
//! ---
//!
//! ## Pattern Matching on Errors
//!
//! Use pattern matching to handle specific error cases precisely.
//!
//! ### Matching ThalesError Variants
//!
//! ```rust
//! use thales::{ThalesError, solver::SolverError};
//!
//! fn handle_thales_error(err: ThalesError) {
//!     match err {
//!         ThalesError::Parse(parse_err) => {
//!             println!("Parse failed: {}", parse_err);
//!         }
//!         ThalesError::Solver(SolverError::NoSolution) => {
//!             println!("No solution exists");
//!         }
//!         ThalesError::Solver(SolverError::InfiniteSolutions) => {
//!             println!("Infinitely many solutions");
//!         }
//!         ThalesError::Numerical(_) => {
//!             println!("Numerical method failed");
//!         }
//!         // Always include wildcard for #[non_exhaustive] enums
//!         _ => println!("Other error: {}", err),
//!     }
//! }
//! ```
//!
//! ### Nested Pattern Matching
//!
//! ```rust
//! use thales::{ThalesError, series::SeriesError};
//!
//! fn detailed_error_handling(err: ThalesError) {
//!     match err {
//!         ThalesError::Series(SeriesError::CannotExpand(msg)) => {
//!             println!("Cannot expand: {}", msg);
//!             // Maybe try numerical approximation instead
//!         }
//!         ThalesError::Series(SeriesError::InvalidOrder(msg)) => {
//!             println!("Invalid order: {}", msg);
//!             // Suggest valid order range
//!         }
//!         _ => println!("Error: {}", err),
//!     }
//! }
//! ```
//!
//! ---
//!
//! ## The ? Operator with Thales Errors
//!
//! The `?` operator provides concise error propagation. It automatically
//! converts errors using the `From` trait.
//!
//! ```rust
//! use thales::{ThalesError, parse_expression, taylor, Variable, Expression};
//!
//! // The ? operator propagates errors up the call stack
//! fn compute_taylor_of_parsed(input: &str) -> Result<Expression, ThalesError> {
//!     // parse_expression returns Result<_, Vec<ParseError>>
//!     let expr = parse_expression(input)
//!         .map_err(|errors| {
//!             ThalesError::from(errors.into_iter().next().unwrap())
//!         })?;
//!
//!     // taylor returns Result<_, SeriesError>
//!     // SeriesError automatically converts to ThalesError
//!     let series = taylor(&expr, &Variable::new("x"), 0.0, 5)?;
//!
//!     Ok(series)
//! }
//! ```
//!
//! **Mixing error types:**
//!
//! ```rust,ignore
//! use thales::{ThalesError, parse_equation, SmartSolver, Solution};
//!
//! fn parse_and_solve(input: &str) -> Result<Vec<Solution>, ThalesError> {
//!     // Multiple operations, each with different error types
//!     let eq = parse_equation(input)
//!         .map_err(|e| ThalesError::from(e.into_iter().next().unwrap()))?;
//!
//!     let solver = SmartSolver;
//!     let solutions = solver.solve(&eq)?;  // SolverError -> ThalesError
//!
//!     Ok(solutions)
//! }
//! ```
//!
//! ---
//!
//! ## Error Recovery Strategies
//!
//! Gracefully handle errors and provide fallback behavior.
//!
//! ### Fallback to Numerical Methods
//!
//! ```rust,ignore
//! use thales::{
//!     Equation, SmartSolver, SmartNumericalSolver,
//!     NumericalConfig, solver::SolverError,
//! };
//!
//! fn solve_with_fallback(eq: &Equation) -> Result<Vec<f64>, String> {
//!     // Try symbolic solver first
//!     let symbolic = SmartSolver;
//!     match symbolic.solve(eq) {
//!         Ok(solutions) => {
//!             // Extract numeric values from solutions
//!             Ok(vec![]) // Placeholder
//!         }
//!         Err(SolverError::CannotSolve(_)) => {
//!             // Fall back to numerical solver
//!             let numerical = SmartNumericalSolver::new(NumericalConfig::default());
//!             numerical.solve(eq, &eq.variables().next().unwrap(), 0.0, 10.0)
//!                 .map(|sol| vec![sol.root])
//!                 .map_err(|e| format!("Both solvers failed: {:?}", e))
//!         }
//!         Err(e) => Err(format!("Solver error: {:?}", e)),
//!     }
//! }
//! ```
//!
//! ### Retry with Adjusted Parameters
//!
//! ```rust,ignore
//! use thales::{taylor, Variable, Expression, series::SeriesError};
//!
//! fn taylor_with_retry(expr: &Expression, x: &Variable) -> Result<Expression, SeriesError> {
//!     // Try different orders if expansion fails
//!     for order in [5, 10, 15, 20] {
//!         match taylor(expr, x, 0.0, order) {
//!             Ok(series) => return Ok(series),
//!             Err(SeriesError::InvalidOrder(_)) => continue,
//!             Err(e) => return Err(e),
//!         }
//!     }
//!
//!     Err(SeriesError::CannotExpand("Failed with all orders".into()))
//! }
//! ```
//!
//! ### Provide User-Friendly Messages
//!
//! ```rust
//! use thales::{ThalesError, solver::SolverError};
//!
//! fn user_friendly_error(err: ThalesError) -> String {
//!     match err {
//!         ThalesError::Parse(e) => {
//!             format!("Could not understand the equation. Please check your syntax: {}", e)
//!         }
//!         ThalesError::Solver(SolverError::NoSolution) => {
//!             "This equation has no solution. It may be inconsistent.".to_string()
//!         }
//!         ThalesError::Solver(SolverError::InfiniteSolutions) => {
//!             "This equation is always true and has infinitely many solutions.".to_string()
//!         }
//!         ThalesError::Numerical(_) => {
//!             "Could not find a numerical solution. Try adjusting the search range.".to_string()
//!         }
//!         _ => format!("An error occurred: {}", err),
//!     }
//! }
//! ```
//!
//! ---
//!
//! ## Logging and Debugging Errors
//!
//! Effective error logging helps diagnose issues in production.
//!
//! ### Basic Error Logging
//!
//! ```rust
//! use thales::{ThalesError, parse_expression};
//!
//! fn log_and_process(input: &str) {
//!     match parse_expression(input) {
//!         Ok(expr) => {
//!             println!("[INFO] Successfully parsed: {:?}", expr);
//!         }
//!         Err(errors) => {
//!             eprintln!("[ERROR] Parse failed for input: '{}'", input);
//!             for (i, err) in errors.iter().enumerate() {
//!                 eprintln!("[ERROR] Parse error {}: {}", i + 1, err);
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! ### Debug vs Display Formatting
//!
//! ```rust
//! use thales::{solver::SolverError, ThalesError};
//!
//! let err = ThalesError::Solver(SolverError::NoSolution);
//!
//! // Display format: user-friendly message
//! println!("Display: {}", err);
//! // Output: "Solver error: NoSolution"
//!
//! // Debug format: detailed structure
//! println!("Debug: {:?}", err);
//! // Output: "Solver(NoSolution)"
//! ```
//!
//! ### Error Source Chain
//!
//! ```rust
//! use std::error::Error;
//! use thales::{ThalesError, parser::ParseError};
//!
//! fn print_error_chain(err: &ThalesError) {
//!     eprintln!("Error: {}", err);
//!
//!     // Walk the error source chain
//!     let mut source = err.source();
//!     let mut level = 1;
//!     while let Some(err) = source {
//!         eprintln!("  Caused by ({}): {}", level, err);
//!         source = err.source();
//!         level += 1;
//!     }
//! }
//! ```
//!
//! ### Integration with Logging Crates
//!
//! ```rust,ignore
//! use log::{error, warn, info};
//! use thales::{ThalesError, parse_equation, SmartSolver};
//!
//! fn solve_with_logging(input: &str) {
//!     info!("Attempting to parse equation: {}", input);
//!
//!     let eq = match parse_equation(input) {
//!         Ok(eq) => {
//!             info!("Successfully parsed equation");
//!             eq
//!         }
//!         Err(errors) => {
//!             error!("Parse failed with {} error(s)", errors.len());
//!             for err in errors {
//!                 error!("  - {}", err);
//!             }
//!             return;
//!         }
//!     };
//!
//!     info!("Attempting to solve equation");
//!     match SmartSolver.solve(&eq) {
//!         Ok(solutions) => {
//!             info!("Found {} solution(s)", solutions.len());
//!         }
//!         Err(e) => {
//!             warn!("Solver failed: {:?}", e);
//!         }
//!     }
//! }
//! ```
//!
//! ---
//!
//! ## Best Practices
//!
//! 1. **Always include wildcard match**: Use `_ =>` for `#[non_exhaustive]` enums
//! 2. **Provide context**: Wrap errors with additional information
//! 3. **Use `?` liberally**: Let errors propagate naturally
//! 4. **Match specific cases**: Handle known errors precisely
//! 5. **Log before propagating**: Capture context at each level
//! 6. **Test error paths**: Write tests for error scenarios
//! 7. **User-friendly messages**: Translate technical errors for users
//!
//! ## See Also
//!
//! - [`crate::ThalesError`] - Unified error type documentation
//! - [`crate::guides::solving_equations`] - Equation solving workflows
//! - [`crate::guides::numerical_methods`] - Fallback numerical strategies
//! - [`std::error::Error`] - Rust's standard error trait
