//! Evaluation, variable extraction, and tree traversal for expressions.

use super::{BinaryOp, Expression, Function, SymbolicConstant, UnaryOp};
use std::collections::HashMap;
use std::collections::HashSet;

impl Expression {
    /// Returns a HashSet of all variable names in the expression.
    ///
    /// Recursively traverses the expression tree to collect all unique variable names.
    /// Variables appearing multiple times are only included once.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // x + y * x
    /// let x = Expression::Variable(Variable::new("x"));
    /// let y = Expression::Variable(Variable::new("y"));
    /// let y_times_x = Expression::Binary(BinaryOp::Mul, Box::new(y), Box::new(x.clone()));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y_times_x));
    ///
    /// let vars = expr.variables();
    /// assert_eq!(vars.len(), 2);
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// ```
    ///
    /// # Returns
    ///
    /// A `HashSet<String>` containing all unique variable names.
    ///
    /// # See Also
    ///
    /// - [`contains_variable`](Expression::contains_variable) - Check if a specific variable is present
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    /// Helper function to recursively collect variables.
    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            Expression::Variable(v) => {
                vars.insert(v.name.clone());
            }
            Expression::Unary(_, expr) => {
                expr.collect_variables(vars);
            }
            Expression::Binary(_, left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Expression::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            Expression::Power(base, exp) => {
                base.collect_variables(vars);
                exp.collect_variables(vars);
            }
            _ => {}
        }
    }

    /// Returns true if expression contains the named variable.
    ///
    /// Recursively searches the expression tree for a variable with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The variable name to search for
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // x + 5
    /// let x = Expression::Variable(Variable::new("x"));
    /// let expr = Expression::Binary(
    ///     BinaryOp::Add,
    ///     Box::new(x),
    ///     Box::new(Expression::Integer(5))
    /// );
    ///
    /// assert!(expr.contains_variable("x"));
    /// assert!(!expr.contains_variable("y"));
    /// ```
    ///
    /// # Returns
    ///
    /// `true` if the variable is found, `false` otherwise.
    ///
    /// # See Also
    ///
    /// - [`variables`](Expression::variables) - Get all variables in the expression
    pub fn contains_variable(&self, name: &str) -> bool {
        match self {
            Expression::Variable(v) => v.name == name,
            Expression::Unary(_, expr) => expr.contains_variable(name),
            Expression::Binary(_, left, right) => {
                left.contains_variable(name) || right.contains_variable(name)
            }
            Expression::Function(_, args) => args.iter().any(|arg| arg.contains_variable(name)),
            Expression::Power(base, exp) => {
                base.contains_variable(name) || exp.contains_variable(name)
            }
            _ => false,
        }
    }

    /// Recursively transform the expression using a mapping function.
    ///
    /// Applies the given function to every node in the expression tree, bottom-up.
    /// The transformation is applied to child nodes first, then to the current node.
    ///
    /// # Type Parameters
    ///
    /// * `F` - Function type that maps `&Expression` to `Expression`
    ///
    /// # Arguments
    ///
    /// * `f` - Transformation function to apply to each node
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // Replace all variables named "x" with constant 10
    /// let x = Expression::Variable(Variable::new("x"));
    /// let y = Expression::Variable(Variable::new("y"));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y));
    ///
    /// let transformed = expr.map(&|e| {
    ///     match e {
    ///         Expression::Variable(v) if v.name == "x" => Expression::Integer(10),
    ///         _ => e.clone()
    ///     }
    /// });
    ///
    /// // Result: 10 + y
    /// assert!(transformed.contains_variable("y"));
    /// assert!(!transformed.contains_variable("x"));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`fold`](Expression::fold) - Accumulate values from the expression tree
    pub fn map<F>(&self, f: &F) -> Expression
    where
        F: Fn(&Expression) -> Expression,
    {
        let mapped = match self {
            Expression::Unary(op, expr) => Expression::Unary(*op, Box::new(expr.map(f))),
            Expression::Binary(op, left, right) => {
                Expression::Binary(*op, Box::new(left.map(f)), Box::new(right.map(f)))
            }
            Expression::Function(func, args) => {
                Expression::Function(func.clone(), args.iter().map(|arg| arg.map(f)).collect())
            }
            Expression::Power(base, exp) => {
                Expression::Power(Box::new(base.map(f)), Box::new(exp.map(f)))
            }
            _ => self.clone(),
        };
        f(&mapped)
    }

    /// Fold/reduce the expression tree.
    ///
    /// Accumulates a value by traversing the expression tree and applying a function
    /// to each node along with the accumulated value. Traversal is depth-first.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Type of the accumulated value
    /// * `F` - Function type that combines accumulated value with each node
    ///
    /// # Arguments
    ///
    /// * `init` - Initial accumulated value
    /// * `f` - Reduction function `fn(accumulator, node) -> accumulator`
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp};
    ///
    /// // Count all nodes in the expression tree
    /// let x = Expression::Variable(Variable::new("x"));
    /// let five = Expression::Integer(5);
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(five));
    ///
    /// let node_count = expr.fold(0, &|count, _node| count + 1);
    /// assert_eq!(node_count, 3); // Binary node + 2 leaf nodes
    /// ```
    ///
    /// # See Also
    ///
    /// - [`map`](Expression::map) - Transform each node in the expression tree
    pub fn fold<T, F>(&self, init: T, f: &F) -> T
    where
        F: Fn(T, &Expression) -> T,
    {
        let acc = f(init, self);
        match self {
            Expression::Unary(_, expr) => expr.fold(acc, f),
            Expression::Binary(_, left, right) => {
                let acc = left.fold(acc, f);
                right.fold(acc, f)
            }
            Expression::Function(_, args) => args.iter().fold(acc, |acc, arg| arg.fold(acc, f)),
            Expression::Power(base, exp) => {
                let acc = base.fold(acc, f);
                exp.fold(acc, f)
            }
            _ => acc,
        }
    }

    /// Evaluate the expression with the given variable values.
    ///
    /// Recursively evaluates the expression tree to produce a single floating-point result.
    /// All variables must have values provided in the `vars` map, otherwise evaluation fails.
    ///
    /// # Arguments
    ///
    /// * `vars` - HashMap mapping variable names to their numeric values
    ///
    /// # Returns
    ///
    /// - `Some(f64)` - The computed result if evaluation succeeds
    /// - `None` - If evaluation fails due to:
    ///   - Missing variable value
    ///   - Division by zero
    ///   - Complex result when real number expected
    ///   - Invalid function argument (e.g., sqrt of negative, ln of negative)
    ///   - Custom function encountered (not evaluable)
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::{Expression, Variable, BinaryOp, Function};
    /// use std::collections::HashMap;
    ///
    /// // Evaluate: x^2 + 2*x + 1 with x = 3
    /// let x = Expression::Variable(Variable::new("x"));
    /// let x_squared = Expression::Power(
    ///     Box::new(x.clone()),
    ///     Box::new(Expression::Integer(2))
    /// );
    /// let two_x = Expression::Binary(
    ///     BinaryOp::Mul,
    ///     Box::new(Expression::Integer(2)),
    ///     Box::new(x.clone())
    /// );
    /// let sum1 = Expression::Binary(BinaryOp::Add, Box::new(x_squared), Box::new(two_x));
    /// let expr = Expression::Binary(BinaryOp::Add, Box::new(sum1), Box::new(Expression::Integer(1)));
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("x".to_string(), 3.0);
    /// assert_eq!(expr.evaluate(&vars), Some(16.0)); // 3^2 + 2*3 + 1 = 16
    ///
    /// // Evaluate function: sqrt(16)
    /// let sqrt_16 = Expression::Function(
    ///     Function::Sqrt,
    ///     vec![Expression::Integer(16)]
    /// );
    /// assert_eq!(sqrt_16.evaluate(&HashMap::new()), Some(4.0));
    /// ```
    ///
    /// # See Also
    ///
    /// - [`simplify`](Expression::simplify) - Symbolic simplification before evaluation
    /// - [`variables`](Expression::variables) - Get all variables that need values
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Option<f64> {
        match self {
            Expression::Integer(n) => Some(*n as f64),
            Expression::Rational(r) => Some(*r.numer() as f64 / *r.denom() as f64),
            Expression::Float(x) => Some(*x),
            Expression::Complex(c) => {
                // Only return real part if imaginary is zero
                if c.im.abs() < 1e-10 {
                    Some(c.re)
                } else {
                    None
                }
            }
            Expression::Constant(c) => match c {
                SymbolicConstant::Pi => Some(std::f64::consts::PI),
                SymbolicConstant::E => Some(std::f64::consts::E),
                SymbolicConstant::I => None, // Imaginary unit cannot be evaluated to real f64
            },
            Expression::Variable(v) => vars.get(&v.name).copied(),
            Expression::Unary(op, expr) => {
                let val = expr.evaluate(vars)?;
                match op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Not => Some(if val == 0.0 { 1.0 } else { 0.0 }),
                    UnaryOp::Abs => Some(val.abs()),
                }
            }
            Expression::Binary(op, left, right) => {
                let left_val = left.evaluate(vars)?;
                let right_val = right.evaluate(vars)?;
                match op {
                    BinaryOp::Add => Some(left_val + right_val),
                    BinaryOp::Sub => Some(left_val - right_val),
                    BinaryOp::Mul => Some(left_val * right_val),
                    BinaryOp::Div => {
                        if right_val.abs() < 1e-10 {
                            None
                        } else {
                            Some(left_val / right_val)
                        }
                    }
                    BinaryOp::Mod => Some(left_val % right_val),
                }
            }
            Expression::Function(func, args) => {
                let arg_vals: Option<Vec<f64>> =
                    args.iter().map(|arg| arg.evaluate(vars)).collect();
                let arg_vals = arg_vals?;

                match func {
                    Function::Sin => Some(arg_vals.get(0)?.sin()),
                    Function::Cos => Some(arg_vals.get(0)?.cos()),
                    Function::Tan => Some(arg_vals.get(0)?.tan()),
                    Function::Asin => Some(arg_vals.get(0)?.asin()),
                    Function::Acos => Some(arg_vals.get(0)?.acos()),
                    Function::Atan => Some(arg_vals.get(0)?.atan()),
                    Function::Atan2 => Some(arg_vals.get(0)?.atan2(*arg_vals.get(1)?)),
                    Function::Sinh => Some(arg_vals.get(0)?.sinh()),
                    Function::Cosh => Some(arg_vals.get(0)?.cosh()),
                    Function::Tanh => Some(arg_vals.get(0)?.tanh()),
                    Function::Exp => Some(arg_vals.get(0)?.exp()),
                    Function::Ln => {
                        let x = *arg_vals.get(0)?;
                        if x > 0.0 {
                            Some(x.ln())
                        } else {
                            None
                        }
                    }
                    Function::Log => {
                        let value = *arg_vals.get(0)?;
                        if arg_vals.len() >= 2 {
                            let base = *arg_vals.get(1)?;
                            if value > 0.0 && base > 0.0 {
                                Some(value.log(base))
                            } else {
                                None
                            }
                        } else if value > 0.0 {
                            // Single-arg log(x) = log10(x)
                            Some(value.log10())
                        } else {
                            None
                        }
                    }
                    Function::Log2 => Some(arg_vals.get(0)?.log2()),
                    Function::Log10 => Some(arg_vals.get(0)?.log10()),
                    Function::Sqrt => Some(arg_vals.get(0)?.sqrt()),
                    Function::Cbrt => Some(arg_vals.get(0)?.cbrt()),
                    Function::Pow => Some(arg_vals.get(0)?.powf(*arg_vals.get(1)?)),
                    Function::Floor => Some(arg_vals.get(0)?.floor()),
                    Function::Ceil => Some(arg_vals.get(0)?.ceil()),
                    Function::Round => Some(arg_vals.get(0)?.round()),
                    Function::Abs => Some(arg_vals.get(0)?.abs()),
                    Function::Sign => Some(arg_vals.get(0)?.signum()),
                    Function::Min => arg_vals.iter().copied().reduce(f64::min),
                    Function::Max => arg_vals.iter().copied().reduce(f64::max),
                    // Re/Im/Conj: real context — Re(x)=x, Im(x)=0, Conj(x)=x
                    Function::Re | Function::Conj => arg_vals.first().copied(),
                    Function::Im => Some(0.0),
                    // Special functions: not evaluable in this f64 path
                    Function::Gamma
                    | Function::LnGamma
                    | Function::Digamma
                    | Function::BetaFn
                    | Function::Erf
                    | Function::Erfc
                    | Function::BesselJ
                    | Function::BesselY
                    | Function::BesselI
                    | Function::BesselK
                    | Function::AiryAi
                    | Function::AiryBi
                    | Function::Zeta
                    | Function::Si
                    | Function::Ci
                    | Function::Ei
                    | Function::Heaviside
                    | Function::DiracDelta => None,
                    Function::Custom(_) => None,
                }
            }
            Expression::Power(base, exp) => {
                let base_val = base.evaluate(vars)?;
                let exp_val = exp.evaluate(vars)?;
                Some(base_val.powf(exp_val))
            }
        }
    }
}
