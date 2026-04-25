//! Display implementations and LaTeX generation for AST types.

use super::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
use std::fmt;

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_precedence(f, 0)
    }
}

impl Expression {
    /// Format the expression with precedence awareness.
    fn fmt_with_precedence(&self, f: &mut fmt::Formatter<'_>, parent_prec: u8) -> fmt::Result {
        match self {
            Expression::Integer(n) => write!(f, "{}", n),
            Expression::Rational(r) => write!(f, "{}/{}", r.numer(), r.denom()),
            Expression::Float(x) => write!(f, "{}", x),
            Expression::Complex(c) => {
                if c.im >= 0.0 {
                    write!(f, "{}+{}i", c.re, c.im)
                } else {
                    write!(f, "{}{}i", c.re, c.im)
                }
            }
            Expression::Constant(c) => write!(f, "{}", c),
            Expression::Variable(v) => write!(f, "{}", v),
            Expression::Unary(UnaryOp::Neg, expr) => {
                write!(f, "-")?;
                expr.fmt_with_precedence(f, 3)
            }
            Expression::Unary(UnaryOp::Not, expr) => {
                write!(f, "!")?;
                expr.fmt_with_precedence(f, 3)
            }
            Expression::Unary(UnaryOp::Abs, expr) => {
                write!(f, "|")?;
                expr.fmt_with_precedence(f, 0)?;
                write!(f, "|")
            }
            Expression::Binary(op, left, right) => {
                let prec = op.precedence();
                let needs_parens = prec < parent_prec;

                if needs_parens {
                    write!(f, "(")?;
                }

                left.fmt_with_precedence(f, prec)?;

                match op {
                    BinaryOp::Add => write!(f, " + ")?,
                    BinaryOp::Sub => write!(f, " - ")?,
                    BinaryOp::Mul => write!(f, " * ")?,
                    BinaryOp::Div => write!(f, " / ")?,
                    BinaryOp::Mod => write!(f, " % ")?,
                }

                // Right side needs higher precedence for left-associativity
                right.fmt_with_precedence(f, prec + 1)?;

                if needs_parens {
                    write!(f, ")")?;
                }

                Ok(())
            }
            Expression::Function(func, args) => {
                write!(f, "{}", func)?;
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    arg.fmt_with_precedence(f, 0)?;
                }
                write!(f, ")")
            }
            Expression::Power(base, exp) => {
                let needs_parens = parent_prec > 3;

                if needs_parens {
                    write!(f, "(")?;
                }

                base.fmt_with_precedence(f, 4)?;
                write!(f, "^")?;
                exp.fmt_with_precedence(f, 4)?;

                if needs_parens {
                    write!(f, ")")?;
                }

                Ok(())
            }
        }
    }

    /// Convert the expression to LaTeX notation.
    ///
    /// Renders the expression as a LaTeX string suitable for mathematical
    /// typesetting. Uses proper LaTeX commands for fractions, roots,
    /// Greek letters, and functions.
    ///
    /// # Examples
    ///
    /// ## Simple Expression
    ///
    /// ```
    /// use thales::ast::{Expression, BinaryOp};
    ///
    /// let expr = Expression::Binary(
    ///     BinaryOp::Div,
    ///     Box::new(Expression::Integer(1)),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// assert_eq!(expr.to_latex(), r"\frac{1}{2}");
    /// ```
    ///
    /// ## Power Expression
    ///
    /// ```
    /// use thales::ast::{Expression, Variable};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    /// let expr = Expression::Power(Box::new(x), Box::new(Expression::Integer(2)));
    /// assert_eq!(expr.to_latex(), "x^{2}");
    /// ```
    ///
    /// ## Symbolic Constant
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let pi = Expression::pi();
    /// assert_eq!(pi.to_latex(), r"\pi");
    /// ```
    ///
    /// ## Square Root
    ///
    /// ```
    /// use thales::ast::{Expression, Function, Variable};
    ///
    /// let x = Expression::Variable(Variable::new("x"));
    /// let sqrt_x = Expression::Function(Function::Sqrt, vec![x]);
    /// assert_eq!(sqrt_x.to_latex(), r"\sqrt{x}");
    /// ```
    pub fn to_latex(&self) -> String {
        self.to_latex_inner(0)
    }

    /// Convert to LaTeX with display mode delimiters.
    ///
    /// Wraps the expression in `\[` and `\]` for display math mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let expr = Expression::Integer(42);
    /// assert_eq!(expr.to_latex_display(), r"\[42\]");
    /// ```
    pub fn to_latex_display(&self) -> String {
        format!(r"\[{}\]", self.to_latex())
    }

    /// Convert to LaTeX with inline mode delimiters.
    ///
    /// Wraps the expression in `$` delimiters for inline math mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let expr = Expression::Integer(42);
    /// assert_eq!(expr.to_latex_inline(), "$42$");
    /// ```
    pub fn to_latex_inline(&self) -> String {
        format!("${}$", self.to_latex())
    }

    /// Internal LaTeX generation with precedence tracking.
    fn to_latex_inner(&self, parent_prec: u8) -> String {
        match self {
            Expression::Integer(n) => {
                if *n < 0 {
                    format!("{{{}}}", n)
                } else {
                    n.to_string()
                }
            }
            Expression::Rational(r) => {
                format!(r"\frac{{{}}}{{{}}}", r.numer(), r.denom())
            }
            Expression::Float(x) => {
                if *x < 0.0 {
                    format!("{{{}}}", x)
                } else {
                    x.to_string()
                }
            }
            Expression::Complex(c) => {
                if c.im >= 0.0 {
                    format!("{}+{}i", c.re, c.im)
                } else {
                    format!("{}{}i", c.re, c.im)
                }
            }
            Expression::Constant(c) => match c {
                SymbolicConstant::Pi => r"\pi".to_string(),
                SymbolicConstant::E => "e".to_string(),
                SymbolicConstant::I => "i".to_string(),
            },
            Expression::Variable(v) => {
                // Convert common variable names to Greek letters
                match v.name.as_str() {
                    "alpha" => r"\alpha".to_string(),
                    "beta" => r"\beta".to_string(),
                    "gamma" => r"\gamma".to_string(),
                    "delta" => r"\delta".to_string(),
                    "epsilon" => r"\epsilon".to_string(),
                    "zeta" => r"\zeta".to_string(),
                    "eta" => r"\eta".to_string(),
                    "theta" => r"\theta".to_string(),
                    "iota" => r"\iota".to_string(),
                    "kappa" => r"\kappa".to_string(),
                    "lambda" => r"\lambda".to_string(),
                    "mu" => r"\mu".to_string(),
                    "nu" => r"\nu".to_string(),
                    "xi" => r"\xi".to_string(),
                    "omicron" => r"\omicron".to_string(),
                    "rho" => r"\rho".to_string(),
                    "sigma" => r"\sigma".to_string(),
                    "tau" => r"\tau".to_string(),
                    "upsilon" => r"\upsilon".to_string(),
                    "phi" => r"\phi".to_string(),
                    "chi" => r"\chi".to_string(),
                    "psi" => r"\psi".to_string(),
                    "omega" => r"\omega".to_string(),
                    // Capital Greek
                    "Gamma" => r"\Gamma".to_string(),
                    "Delta" => r"\Delta".to_string(),
                    "Theta" => r"\Theta".to_string(),
                    "Lambda" => r"\Lambda".to_string(),
                    "Xi" => r"\Xi".to_string(),
                    "Pi" => r"\Pi".to_string(),
                    "Sigma" => r"\Sigma".to_string(),
                    "Phi" => r"\Phi".to_string(),
                    "Psi" => r"\Psi".to_string(),
                    "Omega" => r"\Omega".to_string(),
                    // Subscripted variables like x_1
                    name if name.contains('_') => {
                        let parts: Vec<&str> = name.splitn(2, '_').collect();
                        if parts.len() == 2 {
                            format!("{}_{{{}}}", parts[0], parts[1])
                        } else {
                            name.to_string()
                        }
                    }
                    name => name.to_string(),
                }
            }
            Expression::Unary(UnaryOp::Neg, expr) => {
                format!("-{}", expr.to_latex_inner(3))
            }
            Expression::Unary(UnaryOp::Not, expr) => {
                format!(r"\lnot {}", expr.to_latex_inner(3))
            }
            Expression::Unary(UnaryOp::Abs, expr) => {
                format!(r"\left|{}\right|", expr.to_latex_inner(0))
            }
            Expression::Binary(BinaryOp::Div, num, denom) => {
                // Use \frac for division
                format!(
                    r"\frac{{{}}}{{{}}}",
                    num.to_latex_inner(0),
                    denom.to_latex_inner(0)
                )
            }
            Expression::Binary(op, left, right) => {
                let prec = op.precedence();
                let needs_parens = prec < parent_prec;

                let left_str = left.to_latex_inner(prec);
                let right_str = right.to_latex_inner(prec + 1);

                let op_str = match op {
                    BinaryOp::Add => " + ",
                    BinaryOp::Sub => " - ",
                    BinaryOp::Mul => r" \cdot ",
                    BinaryOp::Div => unreachable!(), // Handled above
                    BinaryOp::Mod => r" \bmod ",
                };

                if needs_parens {
                    format!(r"\left({}{}{}right)", left_str, op_str, right_str)
                } else {
                    format!("{}{}{}", left_str, op_str, right_str)
                }
            }
            Expression::Function(func, args) => {
                let func_name = match func {
                    Function::Sin => r"\sin",
                    Function::Cos => r"\cos",
                    Function::Tan => r"\tan",
                    Function::Asin => r"\arcsin",
                    Function::Acos => r"\arccos",
                    Function::Atan => r"\arctan",
                    Function::Atan2 => r"\arctan",
                    Function::Sinh => r"\sinh",
                    Function::Cosh => r"\cosh",
                    Function::Tanh => r"\tanh",
                    Function::Exp => r"\exp",
                    Function::Ln => r"\ln",
                    Function::Log2 => r"\log_2",
                    Function::Log10 => r"\log_{10}",
                    Function::Sqrt => {
                        // Special case: sqrt uses \sqrt{...}
                        if args.len() == 1 {
                            return format!(r"\sqrt{{{}}}", args[0].to_latex_inner(0));
                        }
                        r"\sqrt"
                    }
                    Function::Cbrt => {
                        // Cube root: \sqrt[3]{...}
                        if args.len() == 1 {
                            return format!(r"\sqrt[3]{{{}}}", args[0].to_latex_inner(0));
                        }
                        r"\sqrt[3]"
                    }
                    Function::Abs => {
                        if args.len() == 1 {
                            return format!(r"\left|{}\right|", args[0].to_latex_inner(0));
                        }
                        r"\text{abs}"
                    }
                    Function::Sign => r"\text{sgn}",
                    Function::Floor => r"\lfloor",
                    Function::Ceil => r"\lceil",
                    Function::Round => r"\text{round}",
                    Function::Min => r"\min",
                    Function::Max => r"\max",
                    Function::Pow => {
                        // pow(base, exp) -> base^{exp}
                        if args.len() == 2 {
                            return format!(
                                "{}^{{{}}}",
                                args[0].to_latex_inner(4),
                                args[1].to_latex_inner(0)
                            );
                        }
                        r"\text{pow}"
                    }
                    Function::Log => {
                        // log(value, base) -> \log_{base}(value)
                        if args.len() == 2 {
                            return format!(
                                r"\log_{{{}}}{{{}}}",
                                args[1].to_latex_inner(0),
                                args[0].to_latex_inner(0)
                            );
                        }
                        r"\log"
                    }
                    Function::Re => r"\operatorname{Re}",
                    Function::Im => r"\operatorname{Im}",
                    Function::Conj => {
                        if args.len() == 1 {
                            return format!(r"\overline{{{}}}", args[0].to_latex_inner(0));
                        }
                        r"\operatorname{Conj}"
                    }
                    Function::Custom(name) => {
                        // Custom functions use \text{name}
                        let args_str: Vec<String> =
                            args.iter().map(|a| a.to_latex_inner(0)).collect();
                        return format!(r"\text{{{}}}\left({}\right)", name, args_str.join(", "));
                    }
                };

                // Special handling for floor and ceil
                if matches!(func, Function::Floor) && args.len() == 1 {
                    return format!(r"\lfloor {} \rfloor", args[0].to_latex_inner(0));
                }
                if matches!(func, Function::Ceil) && args.len() == 1 {
                    return format!(r"\lceil {} \rceil", args[0].to_latex_inner(0));
                }

                // Standard function call
                let args_str: Vec<String> = args.iter().map(|a| a.to_latex_inner(0)).collect();
                format!(r"{}\left({}\right)", func_name, args_str.join(", "))
            }
            Expression::Power(base, exp) => {
                let base_str = base.to_latex_inner(4);
                let exp_str = exp.to_latex_inner(0);

                // Add braces around base if it's complex
                let base_needs_braces = matches!(
                    **base,
                    Expression::Binary(_, _, _) | Expression::Unary(_, _)
                );

                if base_needs_braces {
                    format!(r"\left({}\right)^{{{}}}", base_str, exp_str)
                } else {
                    format!("{}^{{{}}}", base_str, exp_str)
                }
            }
        }
    }
}
