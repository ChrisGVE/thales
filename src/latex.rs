//! LaTeX expression parser for mathematical notation.
//!
//! This module provides a parser for LaTeX mathematical expressions, converting
//! LaTeX input strings into the internal [`Expression`] AST for evaluation and
//! manipulation.
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
//! let expr = parse_latex(r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}").unwrap();
//! // Parses the quadratic formula
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

use crate::ast::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp, Variable};
use chumsky::prelude::*;
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
                write!(f, "Unexpected end of input at position {}: expected {}", pos, expected)
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

/// Map of Greek letter commands to their names for variable creation.
fn greek_letter_to_name(cmd: &str) -> Option<&'static str> {
    match cmd {
        "alpha" => Some("alpha"),
        "beta" => Some("beta"),
        "gamma" => Some("gamma"),
        "delta" => Some("delta"),
        "epsilon" => Some("epsilon"),
        "zeta" => Some("zeta"),
        "eta" => Some("eta"),
        "theta" => Some("theta"),
        "iota" => Some("iota"),
        "kappa" => Some("kappa"),
        "lambda" => Some("lambda"),
        "mu" => Some("mu"),
        "nu" => Some("nu"),
        "xi" => Some("xi"),
        "omicron" => Some("omicron"),
        "rho" => Some("rho"),
        "sigma" => Some("sigma"),
        "tau" => Some("tau"),
        "upsilon" => Some("upsilon"),
        "phi" => Some("phi"),
        "chi" => Some("chi"),
        "psi" => Some("psi"),
        "omega" => Some("omega"),
        // Capital Greek letters
        "Alpha" => Some("Alpha"),
        "Beta" => Some("Beta"),
        "Gamma" => Some("Gamma"),
        "Delta" => Some("Delta"),
        "Epsilon" => Some("Epsilon"),
        "Zeta" => Some("Zeta"),
        "Eta" => Some("Eta"),
        "Theta" => Some("Theta"),
        "Iota" => Some("Iota"),
        "Kappa" => Some("Kappa"),
        "Lambda" => Some("Lambda"),
        "Mu" => Some("Mu"),
        "Nu" => Some("Nu"),
        "Xi" => Some("Xi"),
        "Omicron" => Some("Omicron"),
        "Rho" => Some("Rho"),
        "Sigma" => Some("Sigma"),
        "Tau" => Some("Tau"),
        "Upsilon" => Some("Upsilon"),
        "Phi" => Some("Phi"),
        "Chi" => Some("Chi"),
        "Psi" => Some("Psi"),
        "Omega" => Some("Omega"),
        // Variants
        "varepsilon" => Some("epsilon"),
        "vartheta" => Some("theta"),
        "varpi" => Some("pi"),
        "varrho" => Some("rho"),
        "varsigma" => Some("sigma"),
        "varphi" => Some("phi"),
        _ => None,
    }
}

/// Create the LaTeX expression parser.
fn latex_expression_parser<'a>() -> impl Parser<'a, &'a str, Expression, extra::Err<Rich<'a, char>>> {
    recursive(|expr| {
        // Parse numbers (integers and floats)
        let number = text::int(10)
            .then(just('.').then(text::digits(10)).or_not())
            .to_slice()
            .map(|s: &str| {
                if s.contains('.') {
                    Expression::Float(s.parse().unwrap_or(0.0))
                } else {
                    Expression::Integer(s.parse().unwrap_or(0))
                }
            });

        // Parse variable identifiers (single letters or multi-char)
        let identifier = text::ident().map(|s: &str| {
            // Check for symbolic constants
            match s {
                "pi" => Expression::Constant(SymbolicConstant::Pi),
                "e" => Expression::Constant(SymbolicConstant::E),
                "i" => Expression::Constant(SymbolicConstant::I),
                _ => Expression::Variable(Variable::new(s)),
            }
        });

        // Parse braced group: {expr}
        let braced_expr = expr
            .clone()
            .delimited_by(just('{'), just('}'))
            .padded();

        // Parse optional bracketed argument: [expr]
        let bracketed_expr = expr
            .clone()
            .delimited_by(just('['), just(']'))
            .padded();

        // Parse \frac{num}{denom}
        let frac = just('\\')
            .ignore_then(just("frac"))
            .ignore_then(braced_expr.clone())
            .then(braced_expr.clone())
            .map(|(num, denom)| {
                Expression::Binary(BinaryOp::Div, Box::new(num), Box::new(denom))
            });

        // Parse \sqrt{x} or \sqrt[n]{x}
        let sqrt = just('\\')
            .ignore_then(just("sqrt"))
            .ignore_then(bracketed_expr.clone().or_not())
            .then(braced_expr.clone())
            .map(|(opt_n, arg)| match opt_n {
                Some(n) => {
                    // \sqrt[n]{x} = x^(1/n)
                    let exponent = Expression::Binary(
                        BinaryOp::Div,
                        Box::new(Expression::Integer(1)),
                        Box::new(n),
                    );
                    Expression::Power(Box::new(arg), Box::new(exponent))
                }
                None => {
                    // \sqrt{x} = sqrt(x)
                    Expression::Function(Function::Sqrt, vec![arg])
                }
            });

        // Parse Greek letters and special constants
        let greek = just('\\')
            .ignore_then(text::ident())
            .try_map(|cmd: &str, span| {
                match cmd {
                    "pi" => Ok(Expression::Constant(SymbolicConstant::Pi)),
                    "infty" => Ok(Expression::Variable(Variable::new("infinity"))),
                    _ => {
                        if let Some(name) = greek_letter_to_name(cmd) {
                            Ok(Expression::Variable(Variable::new(name)))
                        } else {
                            Err(Rich::custom(span, format!("Unknown Greek letter: \\{}", cmd)))
                        }
                    }
                }
            });

        // Parse LaTeX functions: \sin{x}, \cos{x}, \tan{x}, \ln{x}, \log{x}, \exp{x}
        let latex_func = just('\\')
            .ignore_then(text::ident())
            .then(
                braced_expr.clone()
                    .or(expr.clone().delimited_by(just('('), just(')')).padded())
                    .or(expr.clone().padded())
            )
            .try_map(|(cmd, arg): (&str, Expression), span| {
                let func = match cmd {
                    "sin" => Some(Function::Sin),
                    "cos" => Some(Function::Cos),
                    "tan" => Some(Function::Tan),
                    // Note: cot, sec, csc not in Function enum yet
                    // "cot" => Some(Function::Cot),
                    // "sec" => Some(Function::Sec),
                    // "csc" => Some(Function::Csc),
                    "arcsin" | "asin" => Some(Function::Asin),
                    "arccos" | "acos" => Some(Function::Acos),
                    "arctan" | "atan" => Some(Function::Atan),
                    "sinh" => Some(Function::Sinh),
                    "cosh" => Some(Function::Cosh),
                    "tanh" => Some(Function::Tanh),
                    "ln" => Some(Function::Ln),
                    "log" => Some(Function::Log10),
                    "exp" => Some(Function::Exp),
                    "abs" => Some(Function::Abs),
                    _ => None,
                };
                match func {
                    Some(f) => Ok(Expression::Function(f, vec![arg])),
                    None => Err(Rich::custom(span, format!("Unknown function: \\{}", cmd))),
                }
            });

        // Primary expressions (atoms)
        let primary = choice((
            frac,
            sqrt,
            latex_func,
            greek,
            number.padded(),
            identifier.padded(),
            expr.clone().delimited_by(just('('), just(')')).padded(),
            expr.clone().delimited_by(just('{'), just('}')).padded(),
        ));

        // Handle subscripts: x_n or x_{12}
        let with_subscript = primary.clone()
            .then(
                just('_')
                    .ignore_then(
                        braced_expr.clone()
                            .or(text::ident().map(|s: &str| Expression::Variable(Variable::new(s))))
                            .or(text::int(10).map(|s: &str| Expression::Integer(s.parse().unwrap_or(0))))
                    )
                    .or_not()
            )
            .map(|(base, subscript)| {
                match subscript {
                    Some(sub) => {
                        // Create subscripted variable name
                        if let Expression::Variable(v) = &base {
                            let name = format!("{}_{}", v.name, sub);
                            Expression::Variable(Variable::new(&name))
                        } else {
                            // For non-variable bases, just ignore subscript
                            base
                        }
                    }
                    None => base,
                }
            });

        // Handle exponents: x^n or x^{2}
        let power = with_subscript.clone()
            .then(
                just('^')
                    .ignore_then(
                        braced_expr.clone()
                            .or(just('-').ignore_then(text::int(10)).map(|s: &str| {
                                Expression::Unary(
                                    UnaryOp::Neg,
                                    Box::new(Expression::Integer(s.parse().unwrap_or(0)))
                                )
                            }))
                            .or(text::int(10).map(|s: &str| Expression::Integer(s.parse().unwrap_or(0))))
                            .or(text::ident().map(|s: &str| Expression::Variable(Variable::new(s))))
                    )
                    .repeated()
                    .collect::<Vec<_>>()
            )
            .map(|(base, exponents)| {
                // Right-associative: x^a^b = x^(a^b)
                exponents.into_iter().rev().fold(base, |acc, exp| {
                    Expression::Power(Box::new(acc), Box::new(exp))
                })
            });

        // Negation: -x
        let negation = just('-')
            .repeated()
            .collect::<Vec<_>>()
            .then(power.clone())
            .map(|(negs, expr)| {
                negs.into_iter().fold(expr, |acc, _| {
                    Expression::Unary(UnaryOp::Neg, Box::new(acc))
                })
            });

        // Implicit multiplication: consecutive terms without operator
        // Note: Only first term can have negation; subsequent terms must not start with -
        // This prevents "x - y" from being parsed as "x * (-y)"
        let implicit_mul = negation.clone()
            .then(power.clone().repeated().collect::<Vec<_>>())
            .map(|(first, rest)| {
                rest.into_iter().fold(first, |acc, curr| {
                    Expression::Binary(BinaryOp::Mul, Box::new(acc), Box::new(curr))
                })
            });

        // Multiplication and division with LaTeX operators
        let mul_op = choice((
            just('*').to(BinaryOp::Mul),
            just('/').to(BinaryOp::Div),
            just('\\').ignore_then(just("cdot")).to(BinaryOp::Mul),
            just('\\').ignore_then(just("times")).to(BinaryOp::Mul),
            just('\\').ignore_then(just("div")).to(BinaryOp::Div),
        ));

        let term = implicit_mul.clone()
            .foldl(
                mul_op.padded().then(implicit_mul).repeated(),
                |left, (op, right)| Expression::Binary(op, Box::new(left), Box::new(right)),
            );

        // Addition and subtraction with LaTeX operators
        let add_op = choice((
            just('+').to(BinaryOp::Add),
            just('-').to(BinaryOp::Sub),
            just('\\').ignore_then(just("pm")).to(BinaryOp::Add), // Treat ± as + for now
        ));

        term.clone()
            .foldl(
                add_op.padded().then(term).repeated(),
                |left, (op, right)| Expression::Binary(op, Box::new(left), Box::new(right)),
            )
    })
}

/// Parse a LaTeX expression string into an [`Expression`] AST.
///
/// This function parses LaTeX mathematical notation into the internal
/// expression representation used by the library.
///
/// # Arguments
///
/// * `input` - A LaTeX expression string (e.g., `\frac{1}{2}`, `\sqrt{x}`)
///
/// # Returns
///
/// * `Ok(Expression)` - Successfully parsed expression
/// * `Err(Vec<LaTeXParseError>)` - List of parsing errors with positions
///
/// # Examples
///
/// ## Basic Fraction
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\frac{1}{2}").unwrap();
/// // Creates: 1 / 2
/// ```
///
/// ## Square Root
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\sqrt{x^2 + 1}").unwrap();
/// // Creates: sqrt(x^2 + 1)
/// ```
///
/// ## Nth Root
///
/// ```
/// use thales::latex::parse_latex;
///
/// let expr = parse_latex(r"\sqrt[3]{8}").unwrap();
/// // Creates: 8^(1/3)
/// ```
///
/// ## Greek Letters
///
/// ```
/// use thales::latex::parse_latex;
/// use thales::ast::{Expression, SymbolicConstant};
///
/// let expr = parse_latex(r"2\pi r").unwrap();
/// // Creates: 2 * π * r
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
    latex_expression_parser()
        .padded()
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
                            Some(ch) => LaTeXParseError::UnexpectedCharacter { pos, found: **ch },
                            None => LaTeXParseError::UnexpectedEndOfInput {
                                pos,
                                expected: "expression".to_string(),
                            },
                        },
                        chumsky::error::RichReason::Custom(msg) => {
                            if msg.starts_with("Unknown Greek letter") {
                                LaTeXParseError::InvalidCommand {
                                    pos,
                                    command: msg.replace("Unknown Greek letter: ", ""),
                                }
                            } else if msg.starts_with("Unknown function") {
                                LaTeXParseError::InvalidCommand {
                                    pos,
                                    command: msg.replace("Unknown function: ", ""),
                                }
                            } else {
                                LaTeXParseError::InvalidExpression {
                                    pos,
                                    message: msg.to_string(),
                                }
                            }
                        }
                        _ => LaTeXParseError::InvalidExpression {
                            pos,
                            message: "Parse error".to_string(),
                        },
                    }
                })
                .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

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
            panic!("Expected division");
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
            panic!("Expected sqrt function");
        }
    }

    #[test]
    fn test_parse_nth_root() {
        let expr = parse_latex(r"\sqrt[3]{8}").unwrap();
        if let Expression::Power(base, exp) = expr {
            assert!(matches!(*base, Expression::Integer(8)));
            if let Expression::Binary(BinaryOp::Div, one, three) = *exp {
                assert!(matches!(*one, Expression::Integer(1)));
                assert!(matches!(*three, Expression::Integer(3)));
            }
        } else {
            panic!("Expected power for nth root");
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
            panic!("Expected sin function");
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
            panic!("Expected implicit multiplication");
        }
    }

    #[test]
    fn test_parse_equation() {
        let (left, right) = parse_latex_equation("x^2 = 4").unwrap();
        assert!(matches!(left, Expression::Power(_, _)));
        assert!(matches!(right, Expression::Integer(4)));
    }
}
