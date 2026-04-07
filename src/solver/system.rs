//! Linear system of equations solver.

use crate::ast::{BinaryOp, Equation, Expression, UnaryOp, Variable};
use std::collections::HashMap;

use super::types::{Solution, SolverError, SolverResult};

/// Result type for system solutions.
#[derive(Debug, Clone)]
pub enum SystemSolution {
    /// Unique solution: each variable has exactly one value.
    Unique(HashMap<Variable, Expression>),
    /// Infinite solutions: variables are expressed in terms of free parameters.
    Infinite {
        /// Variables that have specific values.
        bound: HashMap<Variable, Expression>,
        /// Variables that are free parameters (can take any value).
        free: Vec<Variable>,
    },
    /// No solution: the system is inconsistent.
    NoSolution,
}

/// A linear system of equations in matrix form Ax = b.
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// Coefficient matrix A (rows × cols where cols = number of variables).
    pub(crate) coefficients: Vec<Vec<f64>>,
    /// Constant vector b.
    pub(crate) constants: Vec<f64>,
    /// Variable names corresponding to columns.
    pub(crate) variables: Vec<Variable>,
}

impl LinearSystem {
    /// Create a linear system from equations and variables.
    ///
    /// Extracts coefficients from linear equations of the form:
    /// a₁x₁ + a₂x₂ + ... + aₙxₙ = b
    ///
    /// # Errors
    ///
    /// Returns an error if any equation is not linear in the given variables.
    pub fn from_equations(equations: &[Equation], variables: &[Variable]) -> SolverResult<Self> {
        let n_eqs = equations.len();
        let n_vars = variables.len();

        if n_eqs == 0 || n_vars == 0 {
            return Err(SolverError::Other("Empty system".to_string()));
        }

        let mut coefficients = Vec::with_capacity(n_eqs);
        let mut constants = Vec::with_capacity(n_eqs);

        for eq in equations {
            // Move everything to the left: left - right = 0
            let combined = Expression::Binary(
                BinaryOp::Sub,
                Box::new(eq.left.clone()),
                Box::new(eq.right.clone()),
            )
            .simplify();

            let (row, constant) = Self::extract_linear_coefficients(&combined, variables)?;
            coefficients.push(row);
            constants.push(-constant); // Move constant to RHS
        }

        Ok(Self {
            coefficients,
            constants,
            variables: variables.to_vec(),
        })
    }

    /// Extract linear coefficients from an expression.
    /// Returns (coefficients for each variable, constant term).
    fn extract_linear_coefficients(
        expr: &Expression,
        variables: &[Variable],
    ) -> SolverResult<(Vec<f64>, f64)> {
        let mut coeffs = vec![0.0; variables.len()];
        let mut constant = 0.0;

        // Collect all additive terms
        let terms = Self::collect_additive_terms(expr);

        for term in terms {
            // Check which variable this term contains
            let mut found_var = false;
            for (i, var) in variables.iter().enumerate() {
                if term.contains_variable(&var.name) {
                    // Extract coefficient
                    let coeff = Self::extract_coefficient(&term, var)?;
                    coeffs[i] += coeff;
                    found_var = true;
                    break;
                }
            }

            if !found_var {
                // This is a constant term
                let empty_vars: HashMap<String, f64> = HashMap::new();
                match term.evaluate(&empty_vars) {
                    Some(val) => constant += val,
                    None => {
                        return Err(SolverError::Other(format!(
                            "Cannot evaluate constant term: {}",
                            term
                        )));
                    }
                }
            }
        }

        Ok((coeffs, constant))
    }

    /// Collect all terms that are added or subtracted.
    fn collect_additive_terms(expr: &Expression) -> Vec<Expression> {
        match expr {
            Expression::Binary(BinaryOp::Add, left, right) => {
                let mut terms = Self::collect_additive_terms(left);
                terms.extend(Self::collect_additive_terms(right));
                terms
            }
            Expression::Binary(BinaryOp::Sub, left, right) => {
                let mut terms = Self::collect_additive_terms(left);
                // Negate right side terms
                for term in Self::collect_additive_terms(right) {
                    terms.push(Expression::Unary(UnaryOp::Neg, Box::new(term)));
                }
                terms
            }
            _ => vec![expr.clone()],
        }
    }

    /// Extract the coefficient of a variable from a term.
    fn extract_coefficient(term: &Expression, var: &Variable) -> SolverResult<f64> {
        match term {
            // Just the variable: coefficient is 1
            Expression::Variable(v) if v.name == var.name => Ok(1.0),

            // Negated variable: coefficient is -1
            Expression::Unary(UnaryOp::Neg, inner) => {
                let inner_coeff = Self::extract_coefficient(inner, var)?;
                Ok(-inner_coeff)
            }

            // Multiplication: c * x or x * c
            Expression::Binary(BinaryOp::Mul, left, right) => {
                let left_has_var = left.contains_variable(&var.name);
                let right_has_var = right.contains_variable(&var.name);

                if left_has_var && right_has_var {
                    return Err(SolverError::Other(format!(
                        "Non-linear term: {} * {} both contain {}",
                        left, right, var.name
                    )));
                }

                if left_has_var {
                    // Right should be the coefficient
                    let empty: HashMap<String, f64> = HashMap::new();
                    let coeff = right.evaluate(&empty).ok_or_else(|| {
                        SolverError::Other(format!("Cannot evaluate coefficient: {}", right))
                    })?;
                    let var_coeff = Self::extract_coefficient(left, var)?;
                    Ok(coeff * var_coeff)
                } else {
                    // Left should be the coefficient
                    let empty: HashMap<String, f64> = HashMap::new();
                    let coeff = left.evaluate(&empty).ok_or_else(|| {
                        SolverError::Other(format!("Cannot evaluate coefficient: {}", left))
                    })?;
                    let var_coeff = Self::extract_coefficient(right, var)?;
                    Ok(coeff * var_coeff)
                }
            }

            // Division: x / c
            Expression::Binary(BinaryOp::Div, left, right) => {
                if right.contains_variable(&var.name) {
                    return Err(SolverError::Other(format!(
                        "Non-linear: variable {} in denominator",
                        var.name
                    )));
                }
                let empty: HashMap<String, f64> = HashMap::new();
                let divisor = right.evaluate(&empty).ok_or_else(|| {
                    SolverError::Other(format!("Cannot evaluate divisor: {}", right))
                })?;
                if divisor.abs() < 1e-15 {
                    return Err(SolverError::DivisionByZero);
                }
                let var_coeff = Self::extract_coefficient(left, var)?;
                Ok(var_coeff / divisor)
            }

            _ => {
                // Check if this term contains the variable at all
                if term.contains_variable(&var.name) {
                    Err(SolverError::Other(format!(
                        "Cannot extract coefficient from: {}",
                        term
                    )))
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Solve the linear system using Gaussian elimination with partial pivoting.
    pub fn solve(&self) -> SolverResult<SystemSolution> {
        let n_eqs = self.coefficients.len();
        let n_vars = self.variables.len();

        // Create augmented matrix [A|b]
        let mut augmented: Vec<Vec<f64>> = self
            .coefficients
            .iter()
            .zip(self.constants.iter())
            .map(|(row, &c)| {
                let mut new_row = row.clone();
                new_row.push(c);
                new_row
            })
            .collect();

        // Forward elimination with partial pivoting
        let mut pivot_row = 0;
        let mut pivot_cols = Vec::new(); // Track which columns had pivots

        for col in 0..n_vars {
            if pivot_row >= n_eqs {
                break;
            }

            // Find pivot (largest absolute value in column)
            let mut max_row = pivot_row;
            let mut max_val = augmented[pivot_row][col].abs();
            for row in (pivot_row + 1)..n_eqs {
                if augmented[row][col].abs() > max_val {
                    max_val = augmented[row][col].abs();
                    max_row = row;
                }
            }

            // Skip column if all values are (near) zero
            if max_val < 1e-15 {
                continue;
            }

            // Swap rows if necessary
            if max_row != pivot_row {
                augmented.swap(pivot_row, max_row);
            }

            // Record pivot column
            pivot_cols.push(col);

            // Eliminate below
            let pivot_val = augmented[pivot_row][col];
            for row in (pivot_row + 1)..n_eqs {
                let factor = augmented[row][col] / pivot_val;
                augmented[row][col] = 0.0;
                for c in (col + 1)..=n_vars {
                    augmented[row][c] -= factor * augmented[pivot_row][c];
                }
            }

            pivot_row += 1;
        }

        let rank = pivot_cols.len();

        // Check for inconsistency: a row like [0 0 0 ... 0 | c] where c != 0
        for row in rank..n_eqs {
            let rhs = augmented[row][n_vars];
            let all_zeros = augmented[row][0..n_vars].iter().all(|&x| x.abs() < 1e-15);
            if all_zeros && rhs.abs() > 1e-15 {
                return Ok(SystemSolution::NoSolution);
            }
        }

        // Back substitution
        if rank == n_vars {
            // Unique solution
            let mut solution_values = vec![0.0; n_vars];

            // Back substitute from bottom to top
            for i in (0..rank).rev() {
                let col = pivot_cols[i];
                let mut sum = augmented[i][n_vars];
                for j in (col + 1)..n_vars {
                    sum -= augmented[i][j] * solution_values[j];
                }
                solution_values[col] = sum / augmented[i][col];
            }

            let mut result = HashMap::new();
            for (i, var) in self.variables.iter().enumerate() {
                let val = solution_values[i];
                // Convert to integer if close to an integer
                let expr = if (val - val.round()).abs() < 1e-10 {
                    Expression::Integer(val.round() as i64)
                } else {
                    Expression::Float(val)
                };
                result.insert(var.clone(), expr);
            }

            Ok(SystemSolution::Unique(result))
        } else {
            // Infinite solutions: some variables are free
            // Identify free variables (columns without pivots)
            let pivot_set: std::collections::HashSet<_> = pivot_cols.iter().cloned().collect();
            let free_cols: Vec<_> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();

            let free_vars: Vec<_> = free_cols
                .iter()
                .map(|&c| self.variables[c].clone())
                .collect();

            // Back substitute to express bound variables in terms of free variables
            // For now, return a simplified result
            let mut bound = HashMap::new();

            // Back substitute for pivot variables
            for i in (0..rank).rev() {
                let col = pivot_cols[i];
                let rhs = augmented[i][n_vars];

                // Build expression: rhs - sum of (coefficient * free_var)
                let mut terms: Vec<Expression> = vec![];

                // Constant term
                if rhs.abs() > 1e-15 {
                    terms.push(if (rhs - rhs.round()).abs() < 1e-10 {
                        Expression::Integer(rhs.round() as i64)
                    } else {
                        Expression::Float(rhs)
                    });
                }

                // Free variable terms (negated because moved to RHS)
                for &free_col in &free_cols {
                    let coeff = -augmented[i][free_col] / augmented[i][col];
                    if coeff.abs() > 1e-15 {
                        let free_var = Expression::Variable(self.variables[free_col].clone());
                        let term = if (coeff - coeff.round()).abs() < 1e-10 {
                            let int_coeff = coeff.round() as i64;
                            if int_coeff == 1 {
                                free_var
                            } else if int_coeff == -1 {
                                Expression::Unary(UnaryOp::Neg, Box::new(free_var))
                            } else {
                                Expression::Binary(
                                    BinaryOp::Mul,
                                    Box::new(Expression::Integer(int_coeff)),
                                    Box::new(free_var),
                                )
                            }
                        } else {
                            Expression::Binary(
                                BinaryOp::Mul,
                                Box::new(Expression::Float(coeff)),
                                Box::new(free_var),
                            )
                        };
                        terms.push(term);
                    }
                }

                // Combine terms
                let expr = if terms.is_empty() {
                    Expression::Integer(0)
                } else if terms.len() == 1 {
                    terms.remove(0)
                } else {
                    let mut result = terms.remove(0);
                    for term in terms {
                        result =
                            Expression::Binary(BinaryOp::Add, Box::new(result), Box::new(term));
                    }
                    result
                };

                // Divide by pivot coefficient if not 1
                let pivot_coeff = augmented[i][col];
                let final_expr = if (pivot_coeff - 1.0).abs() < 1e-15 {
                    expr
                } else {
                    Expression::Binary(
                        BinaryOp::Div,
                        Box::new(expr),
                        Box::new(if (pivot_coeff - pivot_coeff.round()).abs() < 1e-10 {
                            Expression::Integer(pivot_coeff.round() as i64)
                        } else {
                            Expression::Float(pivot_coeff)
                        }),
                    )
                };

                bound.insert(self.variables[col].clone(), final_expr);
            }

            Ok(SystemSolution::Infinite {
                bound,
                free: free_vars,
            })
        }
    }

    /// Solve a 2x2 or 3x3 system using Cramer's rule.
    pub fn solve_cramers(&self) -> SolverResult<SystemSolution> {
        let n = self.variables.len();
        if self.coefficients.len() != n {
            return Err(SolverError::Other(
                "Cramer's rule requires square system".to_string(),
            ));
        }

        if n != 2 && n != 3 {
            return Err(SolverError::Other(
                "Cramer's rule only implemented for 2x2 and 3x3 systems".to_string(),
            ));
        }

        let det_a = if n == 2 {
            Self::det_2x2(&self.coefficients)
        } else {
            Self::det_3x3(&self.coefficients)
        };

        if det_a.abs() < 1e-15 {
            // Determinant is zero - system may have no or infinite solutions
            // Fall back to Gaussian elimination
            return self.solve();
        }

        let mut result = HashMap::new();

        for i in 0..n {
            // Create matrix with column i replaced by constants
            let mut modified: Vec<Vec<f64>> = self.coefficients.clone();
            for (row, &c) in self.constants.iter().enumerate() {
                modified[row][i] = c;
            }

            let det_i = if n == 2 {
                Self::det_2x2(&modified)
            } else {
                Self::det_3x3(&modified)
            };

            let val = det_i / det_a;
            let expr = if (val - val.round()).abs() < 1e-10 {
                Expression::Integer(val.round() as i64)
            } else {
                Expression::Float(val)
            };

            result.insert(self.variables[i].clone(), expr);
        }

        Ok(SystemSolution::Unique(result))
    }

    /// Compute 2x2 determinant.
    fn det_2x2(m: &[Vec<f64>]) -> f64 {
        m[0][0] * m[1][1] - m[0][1] * m[1][0]
    }

    /// Compute 3x3 determinant using expansion by first row.
    fn det_3x3(m: &[Vec<f64>]) -> f64 {
        let a = m[0][0];
        let b = m[0][1];
        let c = m[0][2];

        let minor1 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let minor2 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
        let minor3 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        a * minor1 - b * minor2 + c * minor3
    }
}

/// System of equations solver.
#[derive(Debug, Default)]
pub struct SystemSolver;

impl SystemSolver {
    /// Creates a new system of equations solver.
    pub fn new() -> Self {
        Self
    }

    /// Solve a system of linear equations for multiple variables.
    ///
    /// Uses Gaussian elimination with partial pivoting for general systems.
    /// For 2x2 and 3x3 systems, Cramer's rule is also available.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::solver::{SystemSolver, SystemSolution};
    /// use thales::ast::{Equation, Expression, Variable, BinaryOp};
    ///
    /// let solver = SystemSolver::new();
    ///
    /// // Solve: x + y = 5, x - y = 1
    /// let x = Variable::new("x");
    /// let y = Variable::new("y");
    ///
    /// let eq1 = Equation::new(
    ///     "eq1",
    ///     Expression::Binary(
    ///         BinaryOp::Add,
    ///         Box::new(Expression::Variable(x.clone())),
    ///         Box::new(Expression::Variable(y.clone())),
    ///     ),
    ///     Expression::Integer(5),
    /// );
    ///
    /// let eq2 = Equation::new(
    ///     "eq2",
    ///     Expression::Binary(
    ///         BinaryOp::Sub,
    ///         Box::new(Expression::Variable(x.clone())),
    ///         Box::new(Expression::Variable(y.clone())),
    ///     ),
    ///     Expression::Integer(1),
    /// );
    ///
    /// let result = solver.solve_linear_system(&[eq1, eq2], &[x.clone(), y.clone()]).unwrap();
    /// match result {
    ///     SystemSolution::Unique(sol) => {
    ///         // x = 3, y = 2
    ///         assert!(sol.contains_key(&x));
    ///         assert!(sol.contains_key(&y));
    ///     }
    ///     _ => panic!("Expected unique solution"),
    /// }
    /// ```
    pub fn solve_linear_system(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        let system = LinearSystem::from_equations(equations, variables)?;
        system.solve()
    }

    /// Solve using Cramer's rule (2x2 and 3x3 systems only).
    pub fn solve_cramers(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<SystemSolution> {
        let system = LinearSystem::from_equations(equations, variables)?;
        system.solve_cramers()
    }

    /// Solve a system of equations for multiple variables.
    ///
    /// This is a legacy method that delegates to solve_linear_system.
    pub fn solve_system(
        &self,
        equations: &[Equation],
        variables: &[Variable],
    ) -> SolverResult<HashMap<Variable, Solution>> {
        let result = self.solve_linear_system(equations, variables)?;

        match result {
            SystemSolution::Unique(sol) => {
                let mut out = HashMap::new();
                for (var, expr) in sol {
                    out.insert(var, Solution::Unique(expr));
                }
                Ok(out)
            }
            SystemSolution::Infinite { bound, free: _ } => {
                let mut out = HashMap::new();
                for (var, expr) in bound {
                    out.insert(
                        var,
                        Solution::Parametric {
                            expression: expr,
                            constraints: vec![],
                        },
                    );
                }
                Ok(out)
            }
            SystemSolution::NoSolution => Err(SolverError::NoSolution),
        }
    }
}
