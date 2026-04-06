//! Expression and equation parser for mathematical notation.
//!
//! This module provides a complete parser for mathematical expressions and equations,
//! converting string input into Abstract Syntax Tree (AST) structures for evaluation
//! and manipulation.
//!
//! # Supported Syntax
//!
//! The parser supports a comprehensive set of mathematical constructs:
//!
//! | Category | Syntax | Example |
//! |----------|--------|---------|
//! | **Numbers** | Integer | `42`, `-17` |
//! | | Float | `3.14`, `-2.5` |
//! | | Scientific notation | `1.5e10`, `2.3e-5` |
//! | **Variables** | Identifiers | `x`, `theta`, `my_var_2` |
//! | **Binary Operators** | Addition | `a + b` |
//! | | Subtraction | `a - b` |
//! | | Multiplication | `a * b` |
//! | | Division | `a / b` |
//! | | Modulo | `a % b` |
//! | | Power | `a ^ b` |
//! | **Unary Operators** | Negation | `-x` |
//! | **Functions** | Trigonometric | `sin(x)`, `cos(x)`, `tan(x)` |
//! | | Inverse trig | `asin(x)`, `acos(x)`, `atan(x)`, `atan2(y, x)` |
//! | | Hyperbolic | `sinh(x)`, `cosh(x)`, `tanh(x)` |
//! | | Exponential | `exp(x)`, `pow(base, exp)` |
//! | | Logarithmic | `ln(x)`, `log(value, base)`, `log2(x)`, `log10(x)` |
//! | | Root | `sqrt(x)`, `cbrt(x)` |
//! | | Rounding | `floor(x)`, `ceil(x)`, `round(x)` |
//! | | Other | `abs(x)`, `sign(x)`, `min(a, b)`, `max(a, b)` |
//! | **Grouping** | Parentheses | `(a + b) * c` |
//! | **Equations** | Equality | `x + 2 = 5` |
//!
//! # Operator Precedence
//!
//! Operators are evaluated in the following order (highest to lowest precedence):
//!
//! 1. **Function calls**: `sin(x)`, `sqrt(y)` - highest precedence
//! 2. **Unary negation**: `-x` - right-associative
//! 3. **Power**: `a ^ b` - right-associative (e.g., `2 ^ 3 ^ 4` = `2 ^ (3 ^ 4)`)
//! 4. **Multiplication/Division/Modulo**: `a * b`, `a / b`, `a % b` - left-associative
//! 5. **Addition/Subtraction**: `a + b`, `a - b` - left-associative
//!
//! Use parentheses to override precedence: `(a + b) * c` evaluates addition before multiplication.
//!
//! # Examples
//!
//! ## Simple Expression
//!
//! ```
//! use thales::parser::parse_expression;
//! use thales::ast::{Expression, BinaryOp};
//!
//! let expr = parse_expression("2 + 3").unwrap();
//! match expr {
//!     Expression::Binary(BinaryOp::Add, _, _) => println!("Parsed addition"),
//!     _ => panic!("Expected addition"),
//! }
//! ```
//!
//! ## Complex Expression with Functions
//!
//! ```
//! use thales::parser::parse_expression;
//!
//! let expr = parse_expression("sin(x) + cos(y) * 2").unwrap();
//! // Parses as: (sin(x)) + ((cos(y)) * 2)
//! ```
//!
//! ## Power Expression (Right-Associative)
//!
//! ```
//! use thales::parser::parse_expression;
//! use thales::ast::Expression;
//!
//! let expr = parse_expression("2 ^ 3 ^ 4").unwrap();
//! // Parses as: 2 ^ (3 ^ 4) = 2 ^ 81, not (2 ^ 3) ^ 4
//! ```
//!
//! ## Equation Parsing
//!
//! ```
//! use thales::parser::parse_equation;
//!
//! let eq = parse_equation("x + 2 = 5").unwrap();
//! println!("Left side: {:?}", eq.left);
//! println!("Right side: {:?}", eq.right);
//! ```
//!
//! ## Error Handling
//!
//! ```
//! use thales::parser::{parse_expression, ParseError};
//!
//! match parse_expression("2 + + 3") {
//!     Ok(expr) => println!("Parsed: {:?}", expr),
//!     Err(errors) => {
//!         for err in errors {
//!             eprintln!("Parse error: {}", err);
//!         }
//!     }
//! }
//! ```
//!
//! # Performance
//!
//! The parser runs in O(n) time complexity where n is the length of the input string.
//! Memory complexity is O(d) where d is the maximum nesting depth of the expression.
//!
//! # Implicit Multiplication
//!
//! The parser supports implicit multiplication in several forms:
//!
//! | Pattern | Interpretation | Example |
//! |---------|----------------|---------|
//! | Number-Variable | Coefficient | `2x` → `2 * x` |
//! | Number-Parenthesis | Coefficient | `2(x+1)` → `2 * (x+1)` |
//! | Parenthesis-Parenthesis | Product | `(a)(b)` → `a * b` |
//! | Variable Variable (spaced) | Product | `x y` → `x * y` |
//!
//! Note: Multi-character identifiers like `xy` or `theta` are NOT split into
//! separate variables. Use spaces to separate variables: `x y` not `xy`.
//!
//! # Symbolic Constants
//!
//! The parser recognizes the following symbolic constants:
//!
//! | Constant | Keyword | Value |
//! |----------|---------|-------|
//! | π (pi) | `pi` | 3.14159... |
//! | e (Euler's number) | `e` | 2.71828... |
//! | i (imaginary unit) | `i` | √(-1) |
//!
//! # Note
//!
//! Parsing is delegated to the [mathlex](https://crates.io/crates/mathlex) library.
//! For LaTeX input, see the [`latex`](crate::latex) module.

use crate::ast::{Equation, Expression};
use crate::mathlex_bridge;

/// Parse error type with detailed position information.
///
/// All error variants include a `pos` field indicating the character position
/// where the error was detected (0-based index).
///
/// # Error Variants
///
/// - **UnexpectedCharacter**: Found a character that doesn't fit the grammar at this position
/// - **UnexpectedEndOfInput**: Input ended when more tokens were expected
/// - **InvalidNumber**: Number format is incorrect (e.g., "1.2.3", "1e")
/// - **UnknownFunction**: Function name not recognized
/// - **MismatchedParentheses**: Opening/closing parentheses don't match
/// - **InvalidExpression**: Generic parse error with custom message
///
/// # Examples
///
/// ```
/// use thales::parser::{parse_expression, ParseError};
///
/// // Unexpected character
/// match parse_expression("2 @ 3") {
///     Err(errors) => {
///         assert!(errors.iter().any(|e| matches!(
///             e,
///             ParseError::UnexpectedCharacter { pos: 2, found: '@' }
///         )));
///     }
///     Ok(_) => panic!("Expected parse error"),
/// }
///
/// // With implicit multiplication, `foo(x)` parses as `foo * x`.
/// // To trigger an actual parse error, use truly invalid syntax:
/// match parse_expression("2 +* 3") {
///     Err(errors) => {
///         assert!(!errors.is_empty());
///     }
///     Ok(_) => panic!("Expected parse error"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ParseError {
    /// Encountered a character that doesn't match the expected grammar.
    UnexpectedCharacter {
        /// Character position in input (0-based)
        pos: usize,
        /// The unexpected character found
        found: char,
    },

    /// Input ended before expression was complete.
    UnexpectedEndOfInput {
        /// Position where input ended (0-based)
        pos: usize,
        /// Description of what was expected
        expected: String,
    },

    /// Number format is invalid (e.g., multiple decimal points).
    InvalidNumber {
        /// Position where number started (0-based)
        pos: usize,
        /// The invalid number text
        text: String,
    },

    /// Function name is not recognized.
    UnknownFunction {
        /// Position where function name started (0-based)
        pos: usize,
        /// The unrecognized function name
        name: String,
    },

    /// Parentheses are not properly matched.
    MismatchedParentheses {
        /// Position of the mismatched parenthesis (0-based)
        pos: usize,
    },

    /// Generic parse error with custom message.
    InvalidExpression {
        /// Position where error occurred (0-based)
        pos: usize,
        /// Detailed error description
        message: String,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedCharacter { pos, found } => {
                write!(f, "Unexpected character '{}' at position {}", found, pos)
            }
            ParseError::UnexpectedEndOfInput { pos, expected } => {
                write!(
                    f,
                    "Unexpected end of input at position {}, expected {}",
                    pos, expected
                )
            }
            ParseError::InvalidNumber { pos, text } => {
                write!(f, "Invalid number '{}' at position {}", text, pos)
            }
            ParseError::UnknownFunction { pos, name } => {
                write!(f, "Unknown function '{}' at position {}", name, pos)
            }
            ParseError::MismatchedParentheses { pos } => {
                write!(f, "Mismatched parentheses at position {}", pos)
            }
            ParseError::InvalidExpression { pos, message } => {
                write!(f, "Invalid expression at position {}: {}", pos, message)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Convert a mathlex ParseError into a thales ParseError.
fn convert_mathlex_error(err: &mathlex::ParseError) -> ParseError {
    let pos = err.span.as_ref().map(|s| s.start.offset).unwrap_or(0);

    use mathlex::ParseErrorKind;
    match &err.kind {
        ParseErrorKind::UnexpectedToken { found, .. } => {
            if let Some(ch) = found.chars().next() {
                ParseError::UnexpectedCharacter { pos, found: ch }
            } else {
                ParseError::InvalidExpression {
                    pos,
                    message: format!("unexpected token: {}", found),
                }
            }
        }
        ParseErrorKind::UnexpectedEof { expected } => ParseError::UnexpectedEndOfInput {
            pos,
            expected: expected.join(", "),
        },
        ParseErrorKind::InvalidNumber { value, reason } => ParseError::InvalidNumber {
            pos,
            text: format!("{}: {}", value, reason),
        },
        ParseErrorKind::UnknownFunction { name } => ParseError::UnknownFunction {
            pos,
            name: name.clone(),
        },
        ParseErrorKind::UnmatchedDelimiter { .. } => ParseError::MismatchedParentheses { pos },
        _ => ParseError::InvalidExpression {
            pos,
            message: format!("{}", err),
        },
    }
}

/// Parses a mathematical expression from string input into an AST.
///
/// This function converts a textual mathematical expression into an Abstract Syntax Tree
/// (AST) represented by the `Expression` enum. The parser supports all operators, functions,
/// and syntax described in the module-level documentation.
///
/// # Arguments
///
/// * `input` - String slice containing the mathematical expression
///
/// # Returns
///
/// * `Ok(Expression)` - Successfully parsed expression as AST
/// * `Err(Vec<ParseError>)` - One or more parse errors with position information
///
/// # Performance
///
/// Runs in O(n) time where n is the input length. Maximum recursion depth is proportional
/// to expression nesting depth.
///
/// # Examples
///
/// ## Simple Arithmetic
///
/// ```
/// use thales::parser::parse_expression;
/// use thales::ast::{Expression, BinaryOp};
///
/// let expr = parse_expression("2 + 3").unwrap();
/// match expr {
///     Expression::Binary(BinaryOp::Add, _, _) => println!("Addition expression"),
///     _ => panic!("Expected addition"),
/// }
/// ```
///
/// ## Scientific Notation
///
/// ```
/// use thales::parser::parse_expression;
/// use thales::ast::Expression;
///
/// let expr = parse_expression("1.5e-10").unwrap();
/// match expr {
///     Expression::Float(val) => assert!((val - 1.5e-10).abs() < 1e-20),
///     _ => panic!("Expected float"),
/// }
/// ```
///
/// ## Nested Functions
///
/// ```
/// use thales::parser::parse_expression;
///
/// let expr = parse_expression("sqrt(abs(-16))").unwrap();
/// // Parses as: sqrt(abs(-16)) = sqrt(16) = 4
/// ```
///
/// ## Operator Precedence
///
/// ```
/// use thales::parser::parse_expression;
///
/// // Multiplication before addition
/// let expr = parse_expression("2 + 3 * 4").unwrap();
/// // Parses as: 2 + (3 * 4) = 14, not (2 + 3) * 4 = 20
///
/// // Power is right-associative
/// let expr = parse_expression("2 ^ 3 ^ 2").unwrap();
/// // Parses as: 2 ^ (3 ^ 2) = 2 ^ 9 = 512, not (2 ^ 3) ^ 2 = 8 ^ 2 = 64
/// ```
///
/// ## Multiple Variables
///
/// ```
/// use thales::parser::parse_expression;
///
/// let expr = parse_expression("x * y + z").unwrap();
/// // Expression with three variables
/// ```
///
/// ## Complex Expression
///
/// ```
/// use thales::parser::parse_expression;
///
/// let expr = parse_expression("sin(x) ^ 2 + cos(x) ^ 2").unwrap();
/// // Trigonometric identity expression
/// ```
///
/// ## Error Handling
///
/// ```
/// use thales::parser::{parse_expression, ParseError};
///
/// // Incomplete expression
/// match parse_expression("2 * ") {
///     Ok(_) => panic!("Should fail"),
///     Err(errors) => {
///         assert!(!errors.is_empty());
///     }
/// }
///
/// // Unary plus is valid: "2 + + 3" parses as 2 + (+3)
/// assert!(parse_expression("2 + + 3").is_ok());
///
/// // Unknown function names are parsed as generic function calls
/// assert!(parse_expression("foo(x)").is_ok());
/// ```
#[must_use = "parsing returns a result that should be used"]
pub fn parse_expression(input: &str) -> Result<Expression, Vec<ParseError>> {
    let ml_expr = mathlex::parse(input).map_err(|e| vec![convert_mathlex_error(&e)])?;
    mathlex_bridge::convert_expression(&ml_expr).map_err(|msg| {
        vec![ParseError::InvalidExpression {
            pos: 0,
            message: msg,
        }]
    })
}

/// Parses a complete equation from string input into an AST.
///
/// An equation consists of two expressions separated by an equals sign (`=`).
/// The parsed equation has an empty ID by default (can be set later using
/// `Equation::with_id()`).
///
/// # Arguments
///
/// * `input` - String slice containing the equation (format: `expression = expression`)
///
/// # Returns
///
/// * `Ok(Equation)` - Successfully parsed equation with left and right sides
/// * `Err(Vec<ParseError>)` - One or more parse errors with position information
///
/// # Performance
///
/// Runs in O(n) time where n is the input length. Equivalent to parsing two expressions
/// plus the equals sign.
///
/// # Examples
///
/// ## Simple Linear Equation
///
/// ```
/// use thales::parser::parse_equation;
///
/// let eq = parse_equation("x + 2 = 5").unwrap();
/// assert_eq!(eq.id, "");
/// // eq.left: x + 2
/// // eq.right: 5
/// ```
///
/// ## Quadratic Equation
///
/// ```
/// use thales::parser::parse_equation;
///
/// let eq = parse_equation("x^2 + 3*x - 4 = 0").unwrap();
/// // Standard form quadratic equation
/// ```
///
/// ## Equation with Functions
///
/// ```
/// use thales::parser::parse_equation;
///
/// let eq = parse_equation("sin(x) = 0.5").unwrap();
/// // Trigonometric equation
/// ```
///
/// ## Complex Equation
///
/// ```
/// use thales::parser::parse_equation;
///
/// let eq = parse_equation("sqrt(x^2 + y^2) = r").unwrap();
/// // Distance formula equation
/// ```
///
/// ## Setting an ID
///
/// ```
/// use thales::parser::parse_equation;
///
/// let mut eq = parse_equation("F = m * a").unwrap();
/// eq.id = "newton_second_law".to_string();
/// assert_eq!(eq.id, "newton_second_law");
/// ```
///
/// ## Error Handling
///
/// ```
/// use thales::parser::parse_equation;
///
/// // Missing equals sign
/// assert!(parse_equation("x + 2").is_err());
///
/// // Multiple equals signs (not supported)
/// assert!(parse_equation("x = y = 5").is_err());
///
/// // Unary plus is valid, so "2 + + 3 = 5" parses successfully
/// assert!(parse_equation("2 + + 3 = 5").is_ok());
///
/// // Invalid expression on either side
/// assert!(parse_equation("x = 2 * * 3").is_err());
/// ```
///
/// # Limitations
///
/// - Only single equations supported (no equation systems)
/// - Exactly one equals sign required
/// - Both sides must be valid expressions
#[must_use = "parsing returns a result that should be used"]
pub fn parse_equation(input: &str) -> Result<Equation, Vec<ParseError>> {
    // Parse as a single equation via mathlex's equation system parser (one equation)
    let ml_expr = mathlex::parse(input).map_err(|e| vec![convert_mathlex_error(&e)])?;
    mathlex_bridge::convert_equation(&ml_expr).map_err(|msg| {
        vec![ParseError::InvalidExpression {
            pos: 0,
            message: msg,
        }]
    })
}

/// Parse a semicolon-separated list of equations into a vector of [`Equation`] values.
///
/// Each segment between semicolons is trimmed and parsed as a single equation using
/// [`parse_equation`]. Empty segments (e.g. trailing semicolons) are silently skipped.
/// If any segment fails to parse, the errors from **all** failing segments are
/// collected and returned together.
///
/// # Arguments
///
/// * `input` - A string containing one or more equations separated by `';'`.
///
/// # Returns
///
/// - `Ok(Vec<Equation>)` — all segments parsed successfully (empty input returns an
///   empty vector).
/// - `Err(Vec<ParseError>)` — one or more segments failed; every error from every
///   failing segment is included.
///
/// # Examples
///
/// ## Two-equation linear system
///
/// ```
/// use thales::parser::parse_equation_system;
///
/// let eqs = parse_equation_system("x + y = 5; 2*x - y = 1").unwrap();
/// assert_eq!(eqs.len(), 2);
/// ```
///
/// ## Trailing semicolon is ignored
///
/// ```
/// use thales::parser::parse_equation_system;
///
/// let eqs = parse_equation_system("x = 1;").unwrap();
/// assert_eq!(eqs.len(), 1);
/// ```
///
/// ## Empty input returns empty vector
///
/// ```
/// use thales::parser::parse_equation_system;
///
/// let eqs = parse_equation_system("").unwrap();
/// assert!(eqs.is_empty());
/// ```
///
/// ## Parse error propagation
///
/// ```
/// use thales::parser::parse_equation_system;
///
/// assert!(parse_equation_system("x + y = 5; not_an_equation").is_err());
/// ```
#[must_use = "parsing returns a result that should be used"]
pub fn parse_equation_system(input: &str) -> Result<Vec<Equation>, Vec<ParseError>> {
    let mut equations = Vec::new();
    let mut all_errors: Vec<ParseError> = Vec::new();

    for segment in input.split(';') {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parse_equation(trimmed) {
            Ok(eq) => equations.push(eq),
            Err(errs) => all_errors.extend(errs),
        }
    }

    if all_errors.is_empty() {
        Ok(equations)
    } else {
        Err(all_errors)
    }
}
