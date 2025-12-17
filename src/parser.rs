//! Expression and equation parser using chumsky.
//!
//! Provides parsing capabilities for mathematical expressions and equations
//! from string input into AST structures.

use crate::ast::{BinaryOp, Equation, Expression, Function, UnaryOp, Variable};
use chumsky::prelude::*;

/// Parse error type with detailed position information.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedCharacter { pos: usize, found: char },
    UnexpectedEndOfInput { pos: usize, expected: String },
    InvalidNumber { pos: usize, text: String },
    UnknownFunction { pos: usize, name: String },
    MismatchedParentheses { pos: usize },
    InvalidExpression { pos: usize, message: String },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedCharacter { pos, found } => {
                write!(f, "Unexpected character '{}' at position {}", found, pos)
            }
            ParseError::UnexpectedEndOfInput { pos, expected } => {
                write!(f, "Unexpected end of input at position {}, expected {}", pos, expected)
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

/// Helper to map function names to Function enum.
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
fn expression_parser<'src>() -> impl Parser<'src, &'src str, Expression, extra::Err<Rich<'src, char>>> {
    recursive(|expr| {
        // Parse numbers (integer, float, scientific notation)
        let number = text::int(10)
            .then(just('.').then(text::digits(10)).or_not())
            .then(
                one_of("eE")
                    .then(one_of("+-").or_not())
                    .then(text::digits(10))
                    .or_not()
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
            .then(one_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_").repeated())
            .to_slice()
            .padded();

        // Parse function calls OR variables (function calls have precedence)
        let identifier_expr = identifier.then(
            expr.clone()
                .separated_by(just(',').padded())
                .collect::<Vec<_>>()
                .delimited_by(just('('), just(')'))
                .or_not()
        ).try_map(|(name, args_opt), span| {
            match args_opt {
                Some(args) => {
                    // It's a function call
                    string_to_function(name)
                        .map(|func| Expression::Function(func, args))
                        .ok_or_else(|| Rich::custom(span, format!("Unknown function: {}", name)))
                }
                None => {
                    // It's a variable
                    Ok(Expression::Variable(Variable::new(name)))
                }
            }
        }).padded();

        // Primary expressions: numbers, identifier expressions, or parenthesized expressions
        let primary = choice((
            number,
            identifier_expr,
            expr.clone().delimited_by(just('('), just(')')),
        ))
        .padded();

        // Unary operators: negation
        let unary = just('-')
            .padded()
            .repeated()
            .foldr(primary, |_op, rhs| {
                Expression::Unary(UnaryOp::Neg, Box::new(rhs))
            });

        // Power operator (right-associative)
        let power = unary.clone()
            .then(
                just('^')
                    .padded()
                    .ignore_then(unary.clone())
                    .repeated()
                    .collect::<Vec<_>>()
            )
            .map(|(first, rest)| {
                if rest.is_empty() {
                    first
                } else {
                    // Build right-associative power chain
                    let result = rest.into_iter().rev().fold(None, |acc, curr| {
                        match acc {
                            None => Some(curr),
                            Some(right) => Some(Expression::Power(Box::new(curr), Box::new(right))),
                        }
                    });
                    match result {
                        Some(right) => Expression::Power(Box::new(first), Box::new(right)),
                        None => first,
                    }
                }
            });

        // Multiplication and division (left-associative)
        let product = power.clone()
            .foldl(
                choice((
                    just('*').padded().to(BinaryOp::Mul),
                    just('/').padded().to(BinaryOp::Div),
                    just('%').padded().to(BinaryOp::Mod),
                ))
                .then(power)
                .repeated(),
                |lhs, (op, rhs)| Expression::Binary(op, Box::new(lhs), Box::new(rhs))
            );

        // Addition and subtraction (left-associative)
        let sum = product.clone()
            .foldl(
                choice((
                    just('+').padded().to(BinaryOp::Add),
                    just('-').padded().to(BinaryOp::Sub),
                ))
                .then(product)
                .repeated(),
                |lhs, (op, rhs)| Expression::Binary(op, Box::new(lhs), Box::new(rhs))
            );

        sum
    })
}

/// Parse a mathematical expression from string input.
///
/// # Examples
///
/// ```
/// use mathsolver_core::parser::parse_expression;
/// use mathsolver_core::ast::{Expression, BinaryOp};
///
/// let expr = parse_expression("2 + 3").unwrap();
/// match expr {
///     Expression::Binary(BinaryOp::Add, _, _) => (),
///     _ => panic!("Expected addition"),
/// }
/// ```
pub fn parse_expression(input: &str) -> Result<Expression, Vec<ParseError>> {
    expression_parser()
        .then_ignore(end())
        .parse(input)
        .into_result()
        .map_err(|errors| {
            errors.into_iter().map(|e| {
                let span = e.span();
                let pos = span.start;

                match e.reason() {
                    chumsky::error::RichReason::ExpectedFound { found, .. } => {
                        match found {
                            Some(ch) => ParseError::UnexpectedCharacter { pos, found: **ch },
                            None => ParseError::UnexpectedEndOfInput {
                                pos,
                                expected: "expression".to_string()
                            },
                        }
                    }
                    chumsky::error::RichReason::Custom(msg) => {
                        ParseError::InvalidExpression { pos, message: msg.to_string() }
                    }
                    _ => ParseError::InvalidExpression {
                        pos,
                        message: "Parse error".to_string()
                    },
                }
            }).collect()
        })
}

/// Parse a complete equation from string input.
///
/// # Examples
///
/// ```
/// use mathsolver_core::parser::parse_equation;
///
/// let eq = parse_equation("x + 2 = 5").unwrap();
/// assert_eq!(eq.id, "");
/// ```
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
            errors.into_iter().map(|e| {
                let span = e.span();
                let pos = span.start;

                match e.reason() {
                    chumsky::error::RichReason::ExpectedFound { found, .. } => {
                        match found {
                            Some(ch) => ParseError::UnexpectedCharacter { pos, found: **ch },
                            None => ParseError::UnexpectedEndOfInput {
                                pos,
                                expected: "equation".to_string()
                            },
                        }
                    }
                    chumsky::error::RichReason::Custom(msg) => {
                        ParseError::InvalidExpression { pos, message: msg.to_string() }
                    }
                    _ => ParseError::InvalidExpression {
                        pos,
                        message: "Parse error".to_string()
                    },
                }
            }).collect()
        })
}

// TODO: Add support for implicit multiplication (2x, xy)
// TODO: Add support for equation systems (multiple equations)
// TODO: Add LaTeX-style input parsing
// TODO: Add MathML input parsing
