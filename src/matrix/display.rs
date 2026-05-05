//! Display and LaTeX output for matrices.
//!
//! Display is an I/O boundary (Rule 2): Arc<Expr> elements decompile to
//! Expression here so that Expression::to_latex can render them. Decompile
//! happens only at this seam, never inside matrix computation.

use std::fmt;

use super::{BracketStyle, MatrixExpr};
use crate::numeric::compile::decompile;

impl MatrixExpr {
    /// Render the matrix as LaTeX.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::matrix::{MatrixExpr, BracketStyle};
    /// use thales::numeric::expr::Expr;
    ///
    /// let m = MatrixExpr::from_expr_elements(vec![
    ///     vec![Expr::int(1), Expr::int(2)],
    ///     vec![Expr::int(3), Expr::int(4)],
    /// ]).unwrap();
    ///
    /// let latex = m.to_latex(BracketStyle::Parentheses);
    /// assert!(latex.contains("pmatrix"));
    /// ```
    pub fn to_latex(&self, style: BracketStyle) -> String {
        let env = match style {
            BracketStyle::Parentheses => "pmatrix",
            BracketStyle::Square => "bmatrix",
            BracketStyle::Curly => "Bmatrix",
            BracketStyle::Determinant => "vmatrix",
            BracketStyle::Norm => "Vmatrix",
            BracketStyle::None => "matrix",
        };

        let mut result = format!("\\begin{{{}}}\n", env);
        for (i, row) in self.elements().iter().enumerate() {
            let row_str: Vec<String> = row.iter().map(|e| decompile(e).to_latex()).collect();
            result.push_str(&row_str.join(" & "));
            if i < self.rows() - 1 {
                result.push_str(" \\\\\n");
            } else {
                result.push('\n');
            }
        }
        result.push_str(&format!("\\end{{{}}}", env));
        result
    }

    /// Render the matrix as LaTeX with default parentheses style.
    pub fn to_latex_default(&self) -> String {
        self.to_latex(BracketStyle::default())
    }
}

impl fmt::Display for MatrixExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.elements().iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "[")?;
            for (j, elem) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}
