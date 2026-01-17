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
//! | | Logarithmic | `ln(x)`, `log(x)`, `log2(x)`, `log10(x)` |
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
//! # Limitations
//!
//! - **No equation systems**: Only single equations supported
//! - **No LaTeX input**: Plain ASCII notation only
//! - **No MathML**: Plain text input only
//!
//! See TODO comments at end of file for planned enhancements.

use crate::ast::{BinaryOp, Equation, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use chumsky::prelude::*;

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
/// // Unknown function
/// match parse_expression("foo(x)") {
///     Err(errors) => {
///         assert!(errors.iter().any(|e| matches!(
///             e,
///             ParseError::InvalidExpression { message, .. } if message.contains("Unknown function")
///         )));
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

/// Maps function name strings to the corresponding Function enum variant.
///
/// This helper function converts textual function names (as they appear in expressions)
/// to their corresponding `Function` enum values. Used internally by the parser to
/// recognize valid function calls.
///
/// # Supported Functions
///
/// ## Trigonometric Functions
/// - `sin`, `cos`, `tan` - Standard trigonometric functions
/// - `asin`, `acos`, `atan` - Inverse trigonometric functions
/// - `atan2` - Two-argument arctangent
///
/// ## Hyperbolic Functions
/// - `sinh`, `cosh`, `tanh` - Hyperbolic trigonometric functions
///
/// ## Exponential and Logarithmic Functions
/// - `exp` - Natural exponential (e^x)
/// - `ln` - Natural logarithm (base e)
/// - `log` - Common logarithm (base 10)
/// - `log2` - Binary logarithm (base 2)
/// - `log10` - Common logarithm (base 10)
///
/// ## Power and Root Functions
/// - `sqrt` - Square root
/// - `cbrt` - Cube root
/// - `pow` - Power (base, exponent)
///
/// ## Rounding Functions
/// - `floor` - Round down to nearest integer
/// - `ceil` - Round up to nearest integer
/// - `round` - Round to nearest integer
///
/// ## Other Functions
/// - `abs` - Absolute value
/// - `sign` - Sign function (-1, 0, or 1)
/// - `min` - Minimum of two values
/// - `max` - Maximum of two values
///
/// # Returns
///
/// - `Some(Function)` if the name matches a known function
/// - `None` if the name is not recognized
///
/// # Examples
///
/// ```
/// # use thales::parser::parse_expression;
/// # use thales::ast::{Expression, Function};
/// // Recognized function
/// let expr = parse_expression("sin(0.5)").unwrap();
/// match expr {
///     Expression::Function(Function::Sin, _) => println!("Parsed sin function"),
///     _ => panic!("Expected sin function"),
/// }
///
/// // Unrecognized function results in parse error
/// assert!(parse_expression("unknown_func(x)").is_err());
/// ```
fn string_to_function(name: &str) -> Option<Function> {
    match name {
        // Trigonometric
        "sin" => Some(Function::Sin),
        "cos" => Some(Function::Cos),
        "tan" => Some(Function::Tan),
        "asin" => Some(Function::Asin),
        "acos" => Some(Function::Acos),
        "atan" => Some(Function::Atan),
        "atan2" => Some(Function::Atan2),

        // Hyperbolic
        "sinh" => Some(Function::Sinh),
        "cosh" => Some(Function::Cosh),
        "tanh" => Some(Function::Tanh),

        // Exponential and logarithmic
        "exp" => Some(Function::Exp),
        "ln" => Some(Function::Ln),
        "log" => Some(Function::Log),
        "log2" => Some(Function::Log2),
        "log10" => Some(Function::Log10),

        // Power and root
        "sqrt" => Some(Function::Sqrt),
        "cbrt" => Some(Function::Cbrt),
        "pow" => Some(Function::Pow),

        // Rounding
        "floor" => Some(Function::Floor),
        "ceil" => Some(Function::Ceil),
        "round" => Some(Function::Round),

        // Other
        "abs" => Some(Function::Abs),
        "sign" => Some(Function::Sign),
        "min" => Some(Function::Min),
        "max" => Some(Function::Max),

        _ => None,
    }
}

/// Build the expression parser.
fn expression_parser<'src>(
) -> impl Parser<'src, &'src str, Expression, extra::Err<Rich<'src, char>>> {
    recursive(|expr| {
        // Parse numbers (integer, float, scientific notation)
        let number = text::int(10)
            .then(just('.').then(text::digits(10)).or_not())
            .then(
                one_of("eE")
                    .then(one_of("+-").or_not())
                    .then(text::digits(10))
                    .or_not(),
            )
            .to_slice()
            .try_map(|s: &str, span| {
                s.parse::<f64>()
                    .map(Expression::Float)
                    .map_err(|_| Rich::custom(span, format!("Invalid number: {}", s)))
            })
            .padded();

        // Parse identifiers (variables or function names)
        // Identifiers start with a letter or underscore, followed by any number of
        // letters, digits, or underscores
        let identifier = one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
            .then(
                one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                    .repeated(),
            )
            .to_slice()
            .padded();

        // Parse function calls, symbolic constants, or variables
        // Priority: function calls > symbolic constants > variables
        let identifier_expr = identifier
            .then(
                expr.clone()
                    .separated_by(just(',').padded())
                    .collect::<Vec<_>>()
                    .delimited_by(just('('), just(')'))
                    .or_not(),
            )
            .try_map(|(name, args_opt), span| {
                match args_opt {
                    Some(args) => {
                        // It's a function call
                        string_to_function(name)
                            .map(|func| Expression::Function(func, args))
                            .ok_or_else(|| {
                                Rich::custom(span, format!("Unknown function: {}", name))
                            })
                    }
                    None => {
                        // Check for symbolic constants first
                        match name {
                            "pi" => Ok(Expression::Constant(SymbolicConstant::Pi)),
                            "e" => Ok(Expression::Constant(SymbolicConstant::E)),
                            "i" => Ok(Expression::Constant(SymbolicConstant::I)),
                            _ => {
                                // It's a variable
                                Ok(Expression::Variable(Variable::new(name)))
                            }
                        }
                    }
                }
            })
            .padded();

        // Clone parsers for later use in implicit multiplication
        let number_for_implicit = number.clone();
        let identifier_for_implicit = identifier_expr.clone();

        // Primary expressions: numbers, identifier expressions, or parenthesized expressions
        let primary = choice((
            number,
            identifier_expr,
            expr.clone().delimited_by(just('('), just(')')),
        ))
        .padded();

        // Unary operators: negation
        let unary = just('-').padded().repeated().foldr(primary, |_op, rhs| {
            Expression::Unary(UnaryOp::Neg, Box::new(rhs))
        });

        // Power operator (right-associative)
        let power = unary
            .clone()
            .then(
                just('^')
                    .padded()
                    .ignore_then(unary.clone())
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .map(|(first, rest)| {
                if rest.is_empty() {
                    first
                } else {
                    // Build right-associative power chain
                    let result = rest.into_iter().rev().fold(None, |acc, curr| match acc {
                        None => Some(curr),
                        Some(right) => Some(Expression::Power(Box::new(curr), Box::new(right))),
                    });
                    match result {
                        Some(right) => Expression::Power(Box::new(first), Box::new(right)),
                        None => first,
                    }
                }
            });

        // Implicit multiplication: handles 2x, xy, 2(x+1), (x)(y)
        // This parses one or more power expressions without operators between them
        // and treats juxtaposition as multiplication.
        // IMPORTANT: We only allow implicit multiplication for expressions that
        // start with a number, identifier, or opening paren - NOT unary minus.
        // This prevents "a - b" from being parsed as "a * (-b)".
        let implicit_mul_operand = choice((
            number_for_implicit,
            identifier_for_implicit,
            expr.clone().delimited_by(just('('), just(')')),
        ))
        .padded()
        .then(
            just('^')
                .padded()
                .ignore_then(unary.clone())
                .repeated()
                .collect::<Vec<_>>(),
        )
        .map(|(first, rest)| {
            if rest.is_empty() {
                first
            } else {
                // Build right-associative power chain
                let result = rest.into_iter().rev().fold(None, |acc, curr| match acc {
                    None => Some(curr),
                    Some(right) => Some(Expression::Power(Box::new(curr), Box::new(right))),
                });
                match result {
                    Some(right) => Expression::Power(Box::new(first), Box::new(right)),
                    None => first,
                }
            }
        });

        let implicit_product = power
            .clone()
            .then(implicit_mul_operand.repeated().collect::<Vec<_>>())
            .map(|(first, rest)| {
                if rest.is_empty() {
                    first
                } else {
                    // Fold multiple juxtaposed expressions into multiplications
                    rest.into_iter().fold(first, |acc, curr| {
                        Expression::Binary(BinaryOp::Mul, Box::new(acc), Box::new(curr))
                    })
                }
            });

        // Multiplication and division (left-associative)
        // Uses implicit_product as the base to include implicit multiplication
        let product = implicit_product.clone().foldl(
            choice((
                just('*').padded().to(BinaryOp::Mul),
                just('/').padded().to(BinaryOp::Div),
                just('%').padded().to(BinaryOp::Mod),
            ))
            .then(implicit_product.clone())
            .repeated(),
            |lhs, (op, rhs)| Expression::Binary(op, Box::new(lhs), Box::new(rhs)),
        );

        // Addition and subtraction (left-associative)
        let sum = product.clone().foldl(
            choice((
                just('+').padded().to(BinaryOp::Add),
                just('-').padded().to(BinaryOp::Sub),
            ))
            .then(product)
            .repeated(),
            |lhs, (op, rhs)| Expression::Binary(op, Box::new(lhs), Box::new(rhs)),
        );

        sum
    })
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
/// // Invalid syntax
/// match parse_expression("2 + + 3") {
///     Ok(_) => panic!("Should fail"),
///     Err(errors) => {
///         for error in errors {
///             eprintln!("Error: {}", error);
///         }
///     }
/// }
///
/// // Unknown function
/// match parse_expression("foo(x)") {
///     Ok(_) => panic!("Should fail"),
///     Err(errors) => {
///         assert!(errors.iter().any(|e| {
///             matches!(e, ParseError::InvalidExpression { message, .. }
///                 if message.contains("Unknown function"))
///         }));
///     }
/// }
///
/// // Incomplete expression
/// match parse_expression("2 * ") {
///     Ok(_) => panic!("Should fail"),
///     Err(errors) => {
///         assert!(!errors.is_empty());
///     }
/// }
/// ```
#[must_use = "parsing returns a result that should be used"]
pub fn parse_expression(input: &str) -> Result<Expression, Vec<ParseError>> {
    expression_parser()
        .then_ignore(end())
        .parse(input)
        .into_result()
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|e| {
                    let span = e.span();
                    let pos = span.start;

                    match e.reason() {
                        chumsky::error::RichReason::ExpectedFound { found, .. } => match found {
                            Some(ch) => ParseError::UnexpectedCharacter { pos, found: **ch },
                            None => ParseError::UnexpectedEndOfInput {
                                pos,
                                expected: "expression".to_string(),
                            },
                        },
                        chumsky::error::RichReason::Custom(msg) => ParseError::InvalidExpression {
                            pos,
                            message: msg.to_string(),
                        },
                    }
                })
                .collect()
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
/// // Invalid expression on either side
/// assert!(parse_equation("2 + + 3 = 5").is_err());
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
    let equation_parser = expression_parser()
        .then_ignore(just('=').padded())
        .then(expression_parser())
        .map(|(left, right)| Equation::new("", left, right))
        .then_ignore(end());

    equation_parser
        .parse(input)
        .into_result()
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|e| {
                    let span = e.span();
                    let pos = span.start;

                    match e.reason() {
                        chumsky::error::RichReason::ExpectedFound { found, .. } => match found {
                            Some(ch) => ParseError::UnexpectedCharacter { pos, found: **ch },
                            None => ParseError::UnexpectedEndOfInput {
                                pos,
                                expected: "equation".to_string(),
                            },
                        },
                        chumsky::error::RichReason::Custom(msg) => ParseError::InvalidExpression {
                            pos,
                            message: msg.to_string(),
                        },
                    }
                })
                .collect()
        })
}

// TODO: Add support for implicit multiplication (2x, xy)
// TODO: Add support for equation systems (multiple equations)
// TODO: Add LaTeX-style input parsing
// TODO: Add MathML input parsing
