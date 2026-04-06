//! LaTeX expression parser for mathematical notation.
//!
//! This module provides a parser for LaTeX mathematical expressions, converting
//! LaTeX input strings into the internal [`Expression`] AST for evaluation and
//! manipulation. Parsing is delegated to the mathlex library.
//!
//! # Supported Syntax
//!
//! The parser supports common LaTeX mathematical constructs:
//!
//! | Category | LaTeX | Example |
//! |----------|-------|---------|
//! | **Fractions** | `\frac{num}{denom}` | `\frac{1}{2}`, `\frac{x+1}{y}` |
//! | **Roots** | `\sqrt{x}` | `\sqrt{2}`, `\sqrt{x+1}` |
//! | | `\sqrt[n]{x}` | `\sqrt[3]{8}`, `\sqrt[n]{x}` |
//! | **Exponents** | `x^{n}` or `x^n` | `x^{2}`, `x^2`, `e^{-x}` |
//! | **Subscripts** | `x_{n}` or `x_n` | `x_{1}`, `x_1`, `x_{12}` |
//! | **Greek Letters** | `\alpha`, `\pi`, etc. | `\alpha`, `\theta`, `\pi` |
//! | **Operators** | `\cdot`, `\times`, `\div` | `a \cdot b`, `2 \times 3` |
//! | **Functions** | `\sin`, `\cos`, `\ln`, etc. | `\sin{x}`, `\cos(\theta)` |
//!
//! # Examples
//!
//! ## Simple Fraction
//!
//! ```
//! use thales::latex::parse_latex;
//! use thales::ast::Expression;
//!
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//! // Parses to: 1 / 2
//! ```
//!
//! ## Square Root
//!
//! ```
//! use thales::latex::parse_latex;
//!
//! let expr = parse_latex(r"\sqrt{x}").unwrap();
//! // Parses to: sqrt(x)
//! ```
//!
//! ## Greek Letters
//!
//! ```
//! use thales::latex::parse_latex;
//! use thales::ast::{Expression, SymbolicConstant};
//!
//! let expr = parse_latex(r"\pi").unwrap();
//! assert!(matches!(expr, Expression::Constant(SymbolicConstant::Pi)));
//! ```
//!
//! ## Complex Expression
//!
//! ```
//! use thales::latex::parse_latex;
//!
//! let expr = parse_latex(r"\frac{-b + \sqrt{b^2 - 4 \cdot a \cdot c}}{2 \cdot a}").unwrap();
//! // Parses the quadratic formula (positive root)
//! ```
//!
//! # Error Handling
//!
//! ```
//! use thales::latex::{parse_latex, LaTeXParseError};
//!
//! match parse_latex(r"\frac{1}") {
//!     Ok(expr) => println!("Parsed: {:?}", expr),
//!     Err(errors) => {
//!         for err in errors {
//!             eprintln!("LaTeX parse error: {}", err);
//!         }
//!     }
//! }
//! ```

use crate::ast::Expression;
use crate::mathlex_bridge;
use std::fmt;

/// Error type for LaTeX parsing failures.
///
/// Provides detailed information about what went wrong during parsing,
/// including the position in the input string where the error occurred.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LaTeXParseError {
    /// An unexpected character was encountered.
    UnexpectedCharacter {
        /// Position in the input string (0-based).
        pos: usize,
        /// The unexpected character found.
        found: char,
    },
    /// Unexpected end of input.
    UnexpectedEndOfInput {
        /// Position where input ended.
        pos: usize,
        /// Description of what was expected.
        expected: String,
    },
    /// Invalid LaTeX command.
    InvalidCommand {
        /// Position of the command.
        pos: usize,
        /// The unrecognized command.
        command: String,
    },
    /// Missing required argument.
    MissingArgument {
        /// Position of the command.
        pos: usize,
        /// The command missing an argument.
        command: String,
    },
    /// General parse error.
    InvalidExpression {
        /// Position of the error.
        pos: usize,
        /// Description of the error.
        message: String,
    },
}

impl fmt::Display for LaTeXParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaTeXParseError::UnexpectedCharacter { pos, found } => {
                write!(f, "Unexpected character '{}' at position {}", found, pos)
            }
            LaTeXParseError::UnexpectedEndOfInput { pos, expected } => {
                write!(
                    f,
                    "Unexpected end of input at position {}: expected {}",
                    pos, expected
                )
            }
            LaTeXParseError::InvalidCommand { pos, command } => {
                write!(f, "Invalid LaTeX command '{}' at position {}", command, pos)
            }
            LaTeXParseError::MissingArgument { pos, command } => {
                write!(f, "Missing argument for '{}' at position {}", command, pos)
            }
            LaTeXParseError::InvalidExpression { pos, message } => {
                write!(f, "Invalid expression at position {}: {}", pos, message)
            }
        }
    }
}

impl std::error::Error for LaTeXParseError {}

/// Convert a mathlex ParseError into a LaTeXParseError.
fn convert_mathlex_error(err: &mathlex::ParseError) -> LaTeXParseError {
    let pos = err.span.as_ref().map(|s| s.start.offset).unwrap_or(0);

    use mathlex::ParseErrorKind;
    match &err.kind {
        ParseErrorKind::UnexpectedToken { found, .. } => {
            if let Some(ch) = found.chars().next() {
                LaTeXParseError::UnexpectedCharacter { pos, found: ch }
            } else {
                LaTeXParseError::InvalidExpression {
                    pos,
                    message: format!("unexpected token: {}", found),
                }
            }
        }
        ParseErrorKind::UnexpectedEof { expected } => LaTeXParseError::UnexpectedEndOfInput {
            pos,
            expected: expected.join(", "),
        },
        ParseErrorKind::InvalidLatexCommand { command } => LaTeXParseError::InvalidCommand {
            pos,
            command: command.clone(),
        },
        ParseErrorKind::UnmatchedDelimiter { .. } => LaTeXParseError::InvalidExpression {
            pos,
            message: "mismatched delimiters".to_string(),
        },
        _ => LaTeXParseError::InvalidExpression {
            pos,
            message: format!("{}", err),
        },
    }
}

/// Parse a LaTeX expression string into an [`Expression`] AST.
///
/// # Arguments
///
/// * `input` - A LaTeX expression string
///
/// # Returns
///
/// * `Ok(Expression)` - Successfully parsed expression
/// * `Err(Vec<LaTeXParseError>)` - List of parsing errors
///
/// # Examples
///
/// ## Fraction
///
/// ```
/// use thales::latex::parse_latex;
/// use thales::ast::{Expression, BinaryOp};
///
/// let expr = parse_latex(r"\frac{1}{2}").unwrap();
/// // Parses to: 1 / 2
/// ```
///
/// ## Square Root with Expression
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\sqrt{x^2 + 1}").unwrap();
/// // Parses to: sqrt(x^2 + 1)
/// ```
///
/// ## Cube Root
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\sqrt[3]{8}").unwrap();
/// // Parses to: 8^(1/3)
/// ```
///
/// ## Greek Letters and Multiplication
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"2 \cdot \pi").unwrap();
/// // Parses to: 2 * pi
/// ```
///
/// ## Trigonometric Functions
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\sin{\theta} + \cos{\theta}").unwrap();
/// // Creates: sin(theta) + cos(theta)
/// ```
#[must_use = "parsing returns a result that should be used"]
pub fn parse_latex(input: &str) -> Result<Expression, Vec<LaTeXParseError>> {
    let ml_expr = mathlex::parse_latex(input).map_err(|e| vec![convert_mathlex_error(&e)])?;
    mathlex_bridge::convert_expression(&ml_expr).map_err(|msg| {
        vec![LaTeXParseError::InvalidExpression {
            pos: 0,
            message: msg,
        }]
    })
}

/// Parse a LaTeX equation string into left and right [`Expression`]s.
///
/// Equations are split on `=` to create two expressions.
///
/// # Arguments
///
/// * `input` - A LaTeX equation string (e.g., `x^2 = 4`)
///
/// # Returns
///
/// * `Ok((Expression, Expression))` - Left and right sides of the equation
/// * `Err(Vec<LaTeXParseError>)` - List of parsing errors
///
/// # Examples
///
/// ```
/// use thales::latex::parse_latex_equation;
///
/// let (left, right) = parse_latex_equation(r"x^2 = 4").unwrap();
/// // left = x^2, right = 4
/// ```
#[must_use = "parsing returns a result that should be used"]
pub fn parse_latex_equation(input: &str) -> Result<(Expression, Expression), Vec<LaTeXParseError>> {
    // Use mathlex to parse as a LaTeX equation system (single equation)
    let ml_expr = mathlex::parse_latex(input).map_err(|e| vec![convert_mathlex_error(&e)])?;

    match &ml_expr {
        mathlex::Expression::Equation { left, right } => {
            let l = mathlex_bridge::convert_expression(left).map_err(|msg| {
                vec![LaTeXParseError::InvalidExpression {
                    pos: 0,
                    message: msg,
                }]
            })?;
            let r = mathlex_bridge::convert_expression(right).map_err(|msg| {
                vec![LaTeXParseError::InvalidExpression {
                    pos: 0,
                    message: msg,
                }]
            })?;
            Ok((l, r))
        }
        _ => {
            // Fallback: split on '=' manually
            let parts: Vec<&str> = input.split('=').collect();
            if parts.len() != 2 {
                return Err(vec![LaTeXParseError::InvalidExpression {
                    pos: 0,
                    message: "Expected exactly one '=' in equation".to_string(),
                }]);
            }
            let left = parse_latex(parts[0].trim())?;
            let right = parse_latex(parts[1].trim())?;
            Ok((left, right))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Function, SymbolicConstant, UnaryOp};

    #[test]
    fn test_parse_number() {
        let expr = parse_latex("42").unwrap();
        assert!(matches!(expr, Expression::Integer(42)));
    }

    #[test]
    fn test_parse_float() {
        let expr = parse_latex("3.14").unwrap();
        if let Expression::Float(f) = expr {
            assert!((f - 3.14).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_latex("x").unwrap();
        if let Expression::Variable(v) = expr {
            assert_eq!(v.name, "x");
        } else {
            panic!("Expected Variable");
        }
    }

    #[test]
    fn test_parse_frac() {
        let expr = parse_latex(r"\frac{1}{2}").unwrap();
        if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
            assert!(matches!(*num, Expression::Integer(1)));
            assert!(matches!(*denom, Expression::Integer(2)));
        } else {
            panic!("Expected division, got: {:?}", expr);
        }
    }

    #[test]
    fn test_parse_sqrt() {
        let expr = parse_latex(r"\sqrt{x}").unwrap();
        if let Expression::Function(Function::Sqrt, args) = expr {
            assert_eq!(args.len(), 1);
            if let Expression::Variable(v) = &args[0] {
                assert_eq!(v.name, "x");
            }
        } else {
            panic!("Expected sqrt function, got: {:?}", expr);
        }
    }

    #[test]
    fn test_parse_greek_pi() {
        let expr = parse_latex(r"\pi").unwrap();
        assert!(matches!(expr, Expression::Constant(SymbolicConstant::Pi)));
    }

    #[test]
    fn test_parse_greek_theta() {
        let expr = parse_latex(r"\theta").unwrap();
        if let Expression::Variable(v) = expr {
            assert_eq!(v.name, "theta");
        } else {
            panic!("Expected Variable theta");
        }
    }

    #[test]
    fn test_parse_power() {
        let expr = parse_latex("x^2").unwrap();
        if let Expression::Power(base, exp) = expr {
            if let Expression::Variable(v) = *base {
                assert_eq!(v.name, "x");
            }
            assert!(matches!(*exp, Expression::Integer(2)));
        } else {
            panic!("Expected power");
        }
    }

    #[test]
    fn test_parse_power_braced() {
        let expr = parse_latex("x^{10}").unwrap();
        if let Expression::Power(_, exp) = expr {
            assert!(matches!(*exp, Expression::Integer(10)));
        } else {
            panic!("Expected power");
        }
    }

    #[test]
    fn test_parse_sin() {
        let expr = parse_latex(r"\sin{x}").unwrap();
        if let Expression::Function(Function::Sin, args) = expr {
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected sin function, got: {:?}", expr);
        }
    }

    #[test]
    fn test_parse_cdot() {
        let expr = parse_latex(r"a \cdot b").unwrap();
        if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
            if let Expression::Variable(v) = *left {
                assert_eq!(v.name, "a");
            }
            if let Expression::Variable(v) = *right {
                assert_eq!(v.name, "b");
            }
        } else {
            panic!("Expected multiplication");
        }
    }

    #[test]
    fn test_parse_times() {
        let expr = parse_latex(r"2 \times 3").unwrap();
        if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
            assert!(matches!(*left, Expression::Integer(2)));
            assert!(matches!(*right, Expression::Integer(3)));
        } else {
            panic!("Expected multiplication");
        }
    }

    #[test]
    fn test_parse_complex_frac() {
        let expr = parse_latex(r"\frac{x + 1}{y - 2}").unwrap();
        if let Expression::Binary(BinaryOp::Div, num, denom) = expr {
            assert!(matches!(*num, Expression::Binary(BinaryOp::Add, _, _)));
            assert!(matches!(*denom, Expression::Binary(BinaryOp::Sub, _, _)));
        } else {
            panic!("Expected division");
        }
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_latex("x + y").unwrap();
        assert!(matches!(expr, Expression::Binary(BinaryOp::Add, _, _)));
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse_latex("x - y").unwrap();
        assert!(matches!(expr, Expression::Binary(BinaryOp::Sub, _, _)));
    }

    #[test]
    fn test_parse_negation() {
        let expr = parse_latex("-x").unwrap();
        assert!(matches!(expr, Expression::Unary(UnaryOp::Neg, _)));
    }

    #[test]
    fn test_parse_implicit_mul() {
        let expr = parse_latex("2x").unwrap();
        if let Expression::Binary(BinaryOp::Mul, left, right) = expr {
            assert!(matches!(*left, Expression::Integer(2)));
            if let Expression::Variable(v) = *right {
                assert_eq!(v.name, "x");
            }
        } else {
            panic!("Expected implicit multiplication, got: {:?}", expr);
        }
    }

    #[test]
    fn test_parse_equation() {
        let (left, right) = parse_latex_equation("x^2 = 4").unwrap();
        assert!(matches!(left, Expression::Power(_, _)));
        assert!(matches!(right, Expression::Integer(4)));
    }
}
