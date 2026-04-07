//! Symbolic differentiation for expressions.

use super::{BinaryOp, Expression, Function, UnaryOp};

impl Expression {
    /// Compute the symbolic derivative of this expression with respect to a variable.
    ///
    /// Performs symbolic differentiation using standard calculus rules. The result
    /// is an exact symbolic expression (not a numerical approximation) that may
    /// benefit from simplification using [`simplify`](Expression::simplify).
    ///
    /// # Differentiation Rules
    ///
    /// ## Basic Rules
    ///
    /// ### Constant Rule
    /// **d/dx\[c\] = 0**
    ///
    /// The derivative of any constant is zero.
    ///
    /// ### Variable Rule
    /// **d/dx\[x\] = 1**, **d/dx\[y\] = 0** (when differentiating with respect to x)
    ///
    /// The derivative of a variable with respect to itself is 1; with respect to any other variable is 0.
    ///
    /// ### Power Rule
    /// **d/dx[x^n] = n·x^(n-1)**
    ///
    /// When n is constant, multiply by the exponent and reduce the power by 1.
    ///
    /// ### Sum Rule
    /// **d/dx[u + v] = du/dx + dv/dx**
    ///
    /// The derivative of a sum equals the sum of the derivatives.
    ///
    /// ### Difference Rule
    /// **d/dx[u - v] = du/dx - dv/dx**
    ///
    /// The derivative of a difference equals the difference of the derivatives.
    ///
    /// ## Product and Quotient Rules
    ///
    /// ### Product Rule
    /// **d/dx[u·v] = u·(dv/dx) + v·(du/dx)**
    ///
    /// The derivative of a product equals: first times derivative of second, plus second times derivative of first.
    ///
    /// ### Quotient Rule
    /// **d/dx[u/v] = (v·du/dx - u·dv/dx) / v²**
    ///
    /// The derivative of a quotient equals: (bottom times derivative of top minus top times derivative of bottom) over bottom squared.
    ///
    /// ## Chain Rule
    /// **d/dx[f(g(x))] = f'(g(x))·g'(x)**
    ///
    /// The derivative of a composition equals: derivative of outer function evaluated at inner, times derivative of inner function.
    /// This rule is automatically applied to all function derivatives below.
    ///
    /// ## Trigonometric Functions
    ///
    /// All trigonometric derivatives include chain rule application:
    ///
    /// - **d/dx[sin(u)] = cos(u)·du/dx**
    /// - **d/dx[cos(u)] = -sin(u)·du/dx**
    /// - **d/dx[tan(u)] = sec²(u)·du/dx = (1/cos²(u))·du/dx**
    ///
    /// ## Inverse Trigonometric Functions
    ///
    /// - **d/dx[asin(u)] = (1/√(1-u²))·du/dx**
    /// - **d/dx[acos(u)] = (-1/√(1-u²))·du/dx**
    /// - **d/dx[atan(u)] = (1/(1+u²))·du/dx**
    ///
    /// ## Hyperbolic Functions
    ///
    /// - **d/dx[sinh(u)] = cosh(u)·du/dx**
    /// - **d/dx[cosh(u)] = sinh(u)·du/dx**
    /// - **d/dx[tanh(u)] = sech²(u)·du/dx = (1/cosh²(u))·du/dx**
    ///
    /// ## Exponential and Logarithmic Functions
    ///
    /// ### Exponential derivatives
    /// - **d/dx[exp(u)] = exp(u)·du/dx**
    /// - **d/dx[a^u] = a^u·ln(a)·du/dx** (where a is constant)
    /// - **d/dx[u^v] = u^v·(v'·ln(u) + v·u'/u)** (general case)
    ///
    /// ### Logarithmic derivatives
    /// - **d/dx[ln(u)] = (1/u)·du/dx**
    /// - **d/dx[log₁₀(u)] = (1/(u·ln(10)))·du/dx**
    /// - **d/dx[log₂(u)] = (1/(u·ln(2)))·du/dx**
    /// - **d/dx[log_b(u)] = (1/(u·ln(b)))·du/dx**
    ///
    /// ## Root Functions
    ///
    /// - **d/dx[√u] = (1/(2√u))·du/dx**
    /// - **d/dx[∛u] = (1/(3u^(2/3)))·du/dx**
    ///
    /// # Arguments
    ///
    /// * `with_respect_to` - Name of the variable to differentiate with respect to
    ///
    /// # Examples
    ///
    /// ## Power Rule Example
    ///
    /// ```
    /// use thales::ast::{Expression, Variable};
    ///
    /// // Differentiate x^3 with respect to x
    /// // d/dx[x^3] = 3·x^2
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_cubed = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(3))
    /// );
    /// let derivative = x_cubed.differentiate("x").simplify();
    /// // Result simplifies to: 3 * x^2
    /// ```
    ///
    /// ## Polynomial Derivative
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // Differentiate 3x^2 + 2x + 1 with respect to x
    /// // d/dx[3x^2 + 2x + 1] = 6x + 2
    /// let x = Expression::Variable(Variable::new("x"));
    ///
    /// // Build 3x^2
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let three_x_squared = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(3)),
    ///     Box::new(x_squared)
    /// );
    ///
    /// // Build 2x
    /// let two_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x.clone())
    /// );
    ///
    /// // Build 3x^2 + 2x
    /// let sum1 = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(three_x_squared),
    ///     Box::new(two_x)
    /// );
    ///
    /// // Build 3x^2 + 2x + 1
    /// let polynomial = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(sum1),
    ///     Box::new(Expression::Integer(1))
    /// );
    ///
    /// // Compute derivative
    /// let derivative = polynomial.differentiate("x").simplify();
    /// // Result: 6x + 2 (after simplification)
    /// ```
    ///
    /// ## Chain Rule Example
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, Function};
    ///
    /// // Differentiate sin(x^2) with respect to x
    /// // d/dx[sin(x^2)] = cos(x^2)·2x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let sin_x_squared = Expression::Function(Function::Sin, vec![x_squared]);
    ///
    /// let derivative = sin_x_squared.differentiate("x");
    /// // Result: cos(x^2) * (2 * x^1 * 1)
    /// // Simplifies to: cos(x^2) * 2x
    /// ```
    ///
    /// ## Product Rule Example
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp, Function};
    ///
    /// // Differentiate x·sin(x) with respect to x
    /// // d/dx[x·sin(x)] = x·cos(x) + sin(x)·1 = x·cos(x) + sin(x)
    /// let x = Expression::Variable(Variable::new("x"));
    /// let sin_x = Expression::Function(Function::Sin, vec![x.clone()]);
    /// let x_times_sin_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(x.clone()),
    ///     Box::new(sin_x)
    /// );
    ///
    /// let derivative = x_times_sin_x.differentiate("x");
    /// // Result: x·cos(x) + sin(x)·1
    /// ```
    ///
    /// ## Exponential Function Example
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp, Function};
    ///
    /// // Differentiate exp(2x) with respect to x
    /// // d/dx[exp(2x)] = exp(2x)·2
    /// let x = Expression::Variable(Variable::new("x"));
    /// let two_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x.clone())
    /// );
    /// let exp_2x = Expression::Function(Function::Exp, vec![two_x]);
    ///
    /// let derivative = exp_2x.differentiate("x");
    /// // Result: exp(2x) * 2
    /// ```
    ///
    /// ## Logarithmic Function Example
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp, Function};
    ///
    /// // Differentiate ln(x^2 + 1) with respect to x
    /// // d/dx[ln(x^2 + 1)] = (1/(x^2 + 1))·2x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let x_squared_plus_1 = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(x_squared),
    ///     Box::new(Expression::Integer(1))
    /// );
    /// let ln_expr = Expression::Function(Function::Ln, vec![x_squared_plus_1]);
    ///
    /// let derivative = ln_expr.differentiate("x");
    /// // Result: (1/(x^2 + 1)) * (2*x)
    /// ```
    ///
    /// # Returns
    ///
    /// A new [`Expression`] representing the symbolic derivative. The result is automatically
    /// simplified during construction but may benefit from additional simplification using
    /// [`simplify`](Expression::simplify) to remove redundant terms like multiplication by 1
    /// or addition of 0.
    ///
    /// # See Also
    ///
    /// - [`simplify`](Expression::simplify) - Simplify the derivative result to remove redundant terms
    /// - [`evaluate`](Expression::evaluate) - Numerically evaluate the derivative at specific variable values
    pub fn differentiate(&self, with_respect_to: &str) -> Expression {
        match self {
            // Constant rule: d/dx[c] = 0 (for any constant including π, e, i)
            Expression::Integer(_)
            | Expression::Rational(_)
            | Expression::Float(_)
            | Expression::Complex(_)
            | Expression::Constant(_) => Expression::Integer(0),

            // Variable rule: d/dx[x] = 1, d/dx[y] = 0
            Expression::Variable(v) => {
                if v.name == with_respect_to {
                    Expression::Integer(1)
                } else {
                    Expression::Integer(0)
                }
            }

            // Unary operations
            Expression::Unary(op, expr) => {
                let inner_derivative = expr.differentiate(with_respect_to);
                match op {
                    // d/dx[-f] = -f'
                    UnaryOp::Neg => Expression::Unary(UnaryOp::Neg, Box::new(inner_derivative)),
                    // d/dx[|f|] = sign(f) * f' (simplified, assumes f != 0)
                    UnaryOp::Abs => {
                        let sign =
                            Expression::Function(Function::Sign, vec![expr.as_ref().clone()]);
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(sign),
                            Box::new(inner_derivative),
                        )
                    }
                    // d/dx[!f] = 0 (logical NOT is discrete)
                    UnaryOp::Not => Expression::Integer(0),
                }
            }

            // Binary operations
            Expression::Binary(op, left, right) => {
                let left_deriv = left.differentiate(with_respect_to);
                let right_deriv = right.differentiate(with_respect_to);

                match op {
                    // Sum rule: d/dx[u + v] = du/dx + dv/dx
                    BinaryOp::Add => Expression::Binary(
                        BinaryOp::Add,
                        Box::new(left_deriv),
                        Box::new(right_deriv),
                    ),

                    // Difference rule: d/dx[u - v] = du/dx - dv/dx
                    BinaryOp::Sub => Expression::Binary(
                        BinaryOp::Sub,
                        Box::new(left_deriv),
                        Box::new(right_deriv),
                    ),

                    // Product rule: d/dx[u * v] = u * dv/dx + v * du/dx
                    BinaryOp::Mul => {
                        let term1 =
                            Expression::Binary(BinaryOp::Mul, left.clone(), Box::new(right_deriv));
                        let term2 =
                            Expression::Binary(BinaryOp::Mul, right.clone(), Box::new(left_deriv));
                        Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2))
                    }

                    // Quotient rule: d/dx[u / v] = (v * du/dx - u * dv/dx) / v^2
                    BinaryOp::Div => {
                        let numerator_term1 =
                            Expression::Binary(BinaryOp::Mul, right.clone(), Box::new(left_deriv));
                        let numerator_term2 =
                            Expression::Binary(BinaryOp::Mul, left.clone(), Box::new(right_deriv));
                        let numerator = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(numerator_term1),
                            Box::new(numerator_term2),
                        );
                        let denominator =
                            Expression::Power(right.clone(), Box::new(Expression::Integer(2)));
                        Expression::Binary(
                            BinaryOp::Div,
                            Box::new(numerator),
                            Box::new(denominator),
                        )
                    }

                    // Modulo: derivative is complex, not commonly needed
                    BinaryOp::Mod => Expression::Integer(0),
                }
            }

            // Power rule with chain rule
            Expression::Power(base, exponent) => {
                let base_has_var = base.contains_variable(with_respect_to);
                let exp_has_var = exponent.contains_variable(with_respect_to);

                if !base_has_var && !exp_has_var {
                    // d/dx[c^d] = 0 (constant)
                    Expression::Integer(0)
                } else if base_has_var && !exp_has_var {
                    // Power rule: d/dx[u^n] = n * u^(n-1) * du/dx
                    let base_deriv = base.differentiate(with_respect_to);
                    let n_minus_1 = Expression::Binary(
                        BinaryOp::Sub,
                        exponent.clone(),
                        Box::new(Expression::Integer(1)),
                    );
                    let power_term = Expression::Power(base.clone(), Box::new(n_minus_1));
                    let scaled =
                        Expression::Binary(BinaryOp::Mul, exponent.clone(), Box::new(power_term));
                    Expression::Binary(BinaryOp::Mul, Box::new(scaled), Box::new(base_deriv))
                } else if !base_has_var && exp_has_var {
                    // Exponential rule: d/dx[a^v] = a^v * ln(a) * dv/dx
                    let exp_deriv = exponent.differentiate(with_respect_to);
                    let ln_base = Expression::Function(Function::Ln, vec![base.as_ref().clone()]);
                    let power_term = Expression::Power(base.clone(), exponent.clone());
                    let scaled =
                        Expression::Binary(BinaryOp::Mul, Box::new(power_term), Box::new(ln_base));
                    Expression::Binary(BinaryOp::Mul, Box::new(scaled), Box::new(exp_deriv))
                } else {
                    // General case: d/dx[u^v] = u^v * (v' * ln(u) + v * u'/u)
                    // This is the full logarithmic differentiation formula
                    let base_deriv = base.differentiate(with_respect_to);
                    let exp_deriv = exponent.differentiate(with_respect_to);

                    let ln_base = Expression::Function(Function::Ln, vec![base.as_ref().clone()]);
                    let term1 =
                        Expression::Binary(BinaryOp::Mul, Box::new(exp_deriv), Box::new(ln_base));

                    let u_prime_over_u =
                        Expression::Binary(BinaryOp::Div, Box::new(base_deriv), base.clone());
                    let term2 = Expression::Binary(
                        BinaryOp::Mul,
                        exponent.clone(),
                        Box::new(u_prime_over_u),
                    );

                    let sum = Expression::Binary(BinaryOp::Add, Box::new(term1), Box::new(term2));
                    let power = Expression::Power(base.clone(), exponent.clone());

                    Expression::Binary(BinaryOp::Mul, Box::new(power), Box::new(sum))
                }
            }

            // Function derivatives with chain rule
            Expression::Function(func, args) => {
                if args.is_empty() {
                    return Expression::Integer(0);
                }

                match func {
                    // Trigonometric functions
                    Function::Sin => {
                        // d/dx[sin(u)] = cos(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cos_u = Expression::Function(Function::Cos, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(cos_u), Box::new(arg_deriv))
                    }

                    Function::Cos => {
                        // d/dx[cos(u)] = -sin(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sin_u = Expression::Function(Function::Sin, vec![arg.clone()]);
                        let neg_sin = Expression::Unary(UnaryOp::Neg, Box::new(sin_u));
                        Expression::Binary(BinaryOp::Mul, Box::new(neg_sin), Box::new(arg_deriv))
                    }

                    Function::Tan => {
                        // d/dx[tan(u)] = sec^2(u) * du/dx = (1/cos^2(u)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cos_u = Expression::Function(Function::Cos, vec![arg.clone()]);
                        let cos_squared =
                            Expression::Power(Box::new(cos_u), Box::new(Expression::Integer(2)));
                        let sec_squared = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(cos_squared),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(sec_squared),
                            Box::new(arg_deriv),
                        )
                    }

                    // Inverse trigonometric functions
                    Function::Asin => {
                        // d/dx[asin(u)] = 1/sqrt(1 - u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_minus_u_sq = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let sqrt_term = Expression::Function(Function::Sqrt, vec![one_minus_u_sq]);
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(sqrt_term),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Acos => {
                        // d/dx[acos(u)] = -1/sqrt(1 - u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_minus_u_sq = Expression::Binary(
                            BinaryOp::Sub,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let sqrt_term = Expression::Function(Function::Sqrt, vec![one_minus_u_sq]);
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(sqrt_term),
                        );
                        let neg_deriv = Expression::Unary(UnaryOp::Neg, Box::new(deriv_factor));
                        Expression::Binary(BinaryOp::Mul, Box::new(neg_deriv), Box::new(arg_deriv))
                    }

                    Function::Atan => {
                        // d/dx[atan(u)] = 1/(1 + u^2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let u_squared = Expression::Power(
                            Box::new(arg.clone()),
                            Box::new(Expression::Integer(2)),
                        );
                        let one_plus_u_sq = Expression::Binary(
                            BinaryOp::Add,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_squared),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(one_plus_u_sq),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Atan2 => {
                        // d/dx[atan2(y, x)] is more complex, not commonly needed for uncertainty propagation
                        Expression::Integer(0)
                    }

                    // Hyperbolic functions
                    Function::Sinh => {
                        // d/dx[sinh(u)] = cosh(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cosh_u = Expression::Function(Function::Cosh, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(cosh_u), Box::new(arg_deriv))
                    }

                    Function::Cosh => {
                        // d/dx[cosh(u)] = sinh(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sinh_u = Expression::Function(Function::Sinh, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(sinh_u), Box::new(arg_deriv))
                    }

                    Function::Tanh => {
                        // d/dx[tanh(u)] = sech^2(u) * du/dx = (1/cosh^2(u)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let cosh_u = Expression::Function(Function::Cosh, vec![arg.clone()]);
                        let cosh_squared =
                            Expression::Power(Box::new(cosh_u), Box::new(Expression::Integer(2)));
                        let sech_squared = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(cosh_squared),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(sech_squared),
                            Box::new(arg_deriv),
                        )
                    }

                    // Exponential and logarithmic functions
                    Function::Exp => {
                        // d/dx[exp(u)] = exp(u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let exp_u = Expression::Function(Function::Exp, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(exp_u), Box::new(arg_deriv))
                    }

                    Function::Ln => {
                        // d/dx[ln(u)] = (1/u) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let one_over_u = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(arg.clone()),
                        );
                        Expression::Binary(BinaryOp::Mul, Box::new(one_over_u), Box::new(arg_deriv))
                    }

                    Function::Log10 => {
                        // d/dx[log10(u)] = 1/(u * ln(10)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let ln_10 =
                            Expression::Function(Function::Ln, vec![Expression::Integer(10)]);
                        let u_times_ln10 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(arg.clone()),
                            Box::new(ln_10),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_times_ln10),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Log2 => {
                        // d/dx[log2(u)] = 1/(u * ln(2)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let ln_2 = Expression::Function(Function::Ln, vec![Expression::Integer(2)]);
                        let u_times_ln2 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(arg.clone()),
                            Box::new(ln_2),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(u_times_ln2),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Log => {
                        // d/dx[log(u, b)] = 1/(u * ln(b)) * du/dx
                        if args.len() >= 2 {
                            let arg = &args[0];
                            let base = &args[1];
                            let arg_deriv = arg.differentiate(with_respect_to);
                            let ln_base = Expression::Function(Function::Ln, vec![base.clone()]);
                            let u_times_lnb = Expression::Binary(
                                BinaryOp::Mul,
                                Box::new(arg.clone()),
                                Box::new(ln_base),
                            );
                            let deriv_factor = Expression::Binary(
                                BinaryOp::Div,
                                Box::new(Expression::Integer(1)),
                                Box::new(u_times_lnb),
                            );
                            Expression::Binary(
                                BinaryOp::Mul,
                                Box::new(deriv_factor),
                                Box::new(arg_deriv),
                            )
                        } else {
                            Expression::Integer(0)
                        }
                    }

                    // Root functions
                    Function::Sqrt => {
                        // d/dx[sqrt(u)] = 1/(2*sqrt(u)) * du/dx = (1/2) * u^(-1/2) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sqrt_u = Expression::Function(Function::Sqrt, vec![arg.clone()]);
                        let two_sqrt_u = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(Expression::Integer(2)),
                            Box::new(sqrt_u),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(two_sqrt_u),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Cbrt => {
                        // d/dx[cbrt(u)] = 1/(3*u^(2/3)) * du/dx
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let two_thirds = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(2)),
                            Box::new(Expression::Integer(3)),
                        );
                        let u_to_2_3 =
                            Expression::Power(Box::new(arg.clone()), Box::new(two_thirds));
                        let three_u_2_3 = Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(Expression::Integer(3)),
                            Box::new(u_to_2_3),
                        );
                        let deriv_factor = Expression::Binary(
                            BinaryOp::Div,
                            Box::new(Expression::Integer(1)),
                            Box::new(three_u_2_3),
                        );
                        Expression::Binary(
                            BinaryOp::Mul,
                            Box::new(deriv_factor),
                            Box::new(arg_deriv),
                        )
                    }

                    Function::Pow => {
                        // pow(u, v) is equivalent to u^v, handle like Power
                        if args.len() >= 2 {
                            let power_expr = Expression::Power(
                                Box::new(args[0].clone()),
                                Box::new(args[1].clone()),
                            );
                            power_expr.differentiate(with_respect_to)
                        } else {
                            Expression::Integer(0)
                        }
                    }

                    // Rounding functions (derivatives are 0 almost everywhere)
                    Function::Floor | Function::Ceil | Function::Round => Expression::Integer(0),

                    // Absolute value and sign
                    Function::Abs => {
                        // d/dx[abs(u)] = sign(u) * du/dx (simplified)
                        let arg = &args[0];
                        let arg_deriv = arg.differentiate(with_respect_to);
                        let sign_u = Expression::Function(Function::Sign, vec![arg.clone()]);
                        Expression::Binary(BinaryOp::Mul, Box::new(sign_u), Box::new(arg_deriv))
                    }

                    Function::Sign => {
                        // Derivative of sign function is 0 almost everywhere
                        Expression::Integer(0)
                    }

                    // Min/Max (derivatives are piecewise, simplified to 0)
                    Function::Min | Function::Max => Expression::Integer(0),

                    // Custom functions - cannot differentiate
                    Function::Custom(_) => Expression::Integer(0),
                }
            }
        }
    }
}
