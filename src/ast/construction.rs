//! Builder patterns and convenience constructors for AST types.

use super::{Expression, SymbolicConstant};

impl Expression {
    /// Create a symbolic Pi (π) constant.
    ///
    /// Returns an expression representing the mathematical constant π (approximately 3.14159).
    /// This is preserved symbolically during manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let pi = Expression::pi();
    /// assert_eq!(format!("{}", pi), "π");
    /// ```
    #[inline]
    pub fn pi() -> Self {
        Expression::Constant(SymbolicConstant::Pi)
    }

    /// Create a symbolic Euler's number (e) constant.
    ///
    /// Returns an expression representing Euler's number e (approximately 2.71828).
    /// This is the base of the natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let e = Expression::euler();
    /// assert_eq!(format!("{}", e), "e");
    /// ```
    #[inline]
    pub fn euler() -> Self {
        Expression::Constant(SymbolicConstant::E)
    }

    /// Create a symbolic imaginary unit (i) constant.
    ///
    /// Returns an expression representing the imaginary unit i, where i² = -1.
    /// This allows symbolic manipulation of complex expressions.
    ///
    /// # Examples
    ///
    /// ```
    /// use thales::ast::Expression;
    ///
    /// let i = Expression::i();
    /// assert_eq!(format!("{}", i), "i");
    /// ```
    #[inline]
    pub fn i() -> Self {
        Expression::Constant(SymbolicConstant::I)
    }
}
