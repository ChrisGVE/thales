//! Unified error type for the thales library.

/// Unified error type for the thales library.
///
/// This enum provides a single error type that encompasses all possible errors
/// that can occur within the library. It wraps error types from individual modules,
/// allowing for consistent error handling across the entire library.
///
/// # Design
///
/// The `#[non_exhaustive]` attribute allows future versions to add new error variants
/// without breaking existing code. Users should always include a wildcard match arm
/// when matching on this type.
///
/// # Examples
///
/// ```rust
/// use thales::{ThalesError, parse_expression};
///
/// match parse_expression("2 + x") {
///     Ok(expr) => println!("Parsed: {:?}", expr),
///     Err(errors) => {
///         // parse_expression returns Vec<ParseError>, not ThalesError
///         println!("Parse errors: {:?}", errors);
///     }
/// }
/// ```
///
/// Future usage with unified error handling:
///
/// ```rust,ignore
/// use thales::ThalesError;
///
/// fn process() -> Result<(), ThalesError> {
///     // Future API will return ThalesError
///     Ok(())
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ThalesError {
    /// Error from the parser module.
    Parse(crate::parser::ParseError),
    /// Error from the solver module.
    Solver(crate::solver::SolverError),
    /// Error from the matrix module.
    Matrix(crate::matrix::MatrixError),
    /// Error from the integration module.
    Integration(crate::integration::IntegrationError),
    /// Error from the numerical module.
    Numerical(crate::numerical::NumericalError),
    /// Error from the limits module.
    Limits(crate::limits::LimitError),
    /// Error from the ODE module.
    ODE(crate::ode::ODEError),
    /// Error from the special functions module.
    SpecialFunction(crate::special::SpecialFunctionError),
    /// Error from the inequality module.
    Inequality(crate::inequality::InequalityError),
    /// Error from the precision module.
    Evaluation(crate::precision::EvalError),
    /// Error from the partial fractions module.
    PartialFractions(crate::partial_fractions::DecomposeError),
    /// Error from the LaTeX parser module.
    LaTeXParse(crate::latex::LaTeXParseError),
    /// Error from the equation system module.
    System(crate::equation_system::SystemError),
    /// Error from the nonlinear system solver.
    NonlinearSystem(crate::equation_system::NonlinearSystemSolverError),
    /// The `api::execute` dispatcher received a command variant that the
    /// v0.8.1 scaffolding does not yet implement. This error disappears once
    /// task T5 (full dispatch) lands.
    ApiNotImplemented,
}

impl std::fmt::Display for ThalesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThalesError::Parse(e) => write!(f, "Parse error: {}", e),
            ThalesError::Solver(e) => write!(f, "Solver error: {:?}", e),
            ThalesError::Matrix(e) => write!(f, "Matrix error: {}", e),
            ThalesError::Integration(e) => write!(f, "Integration error: {}", e),
            ThalesError::Numerical(e) => write!(f, "Numerical error: {:?}", e),
            ThalesError::Limits(e) => write!(f, "Limits error: {}", e),
            ThalesError::ODE(e) => write!(f, "ODE error: {}", e),
            ThalesError::SpecialFunction(e) => write!(f, "Special function error: {}", e),
            ThalesError::Inequality(e) => write!(f, "Inequality error: {}", e),
            ThalesError::Evaluation(e) => write!(f, "Evaluation error: {}", e),
            ThalesError::PartialFractions(e) => write!(f, "Partial fractions error: {}", e),
            ThalesError::LaTeXParse(e) => write!(f, "LaTeX parse error: {}", e),
            ThalesError::System(e) => write!(f, "System error: {:?}", e),
            ThalesError::NonlinearSystem(e) => write!(f, "Nonlinear system error: {:?}", e),
            ThalesError::ApiNotImplemented => {
                write!(f, "api::execute: command variant not yet implemented")
            }
        }
    }
}

impl std::error::Error for ThalesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ThalesError::Parse(e) => Some(e),
            ThalesError::Matrix(e) => Some(e),
            ThalesError::Integration(e) => Some(e),
            ThalesError::Limits(e) => Some(e),
            ThalesError::ODE(e) => Some(e),
            ThalesError::SpecialFunction(e) => Some(e),
            ThalesError::Inequality(e) => Some(e),
            ThalesError::Evaluation(e) => Some(e),
            ThalesError::PartialFractions(e) => Some(e),
            ThalesError::LaTeXParse(e) => Some(e),
            // These error types don't implement std::error::Error
            ThalesError::Solver(_) => None,
            ThalesError::Numerical(_) => None,
            ThalesError::System(_) => None,
            ThalesError::NonlinearSystem(_) => None,
            ThalesError::ApiNotImplemented => None,
        }
    }
}

impl From<crate::parser::ParseError> for ThalesError {
    fn from(e: crate::parser::ParseError) -> Self {
        ThalesError::Parse(e)
    }
}

impl From<crate::solver::SolverError> for ThalesError {
    fn from(e: crate::solver::SolverError) -> Self {
        ThalesError::Solver(e)
    }
}

impl From<crate::matrix::MatrixError> for ThalesError {
    fn from(e: crate::matrix::MatrixError) -> Self {
        ThalesError::Matrix(e)
    }
}

impl From<crate::integration::IntegrationError> for ThalesError {
    fn from(e: crate::integration::IntegrationError) -> Self {
        ThalesError::Integration(e)
    }
}

impl From<crate::numerical::NumericalError> for ThalesError {
    fn from(e: crate::numerical::NumericalError) -> Self {
        ThalesError::Numerical(e)
    }
}

impl From<crate::limits::LimitError> for ThalesError {
    fn from(e: crate::limits::LimitError) -> Self {
        ThalesError::Limits(e)
    }
}

impl From<crate::ode::ODEError> for ThalesError {
    fn from(e: crate::ode::ODEError) -> Self {
        ThalesError::ODE(e)
    }
}

impl From<crate::special::SpecialFunctionError> for ThalesError {
    fn from(e: crate::special::SpecialFunctionError) -> Self {
        ThalesError::SpecialFunction(e)
    }
}

impl From<crate::inequality::InequalityError> for ThalesError {
    fn from(e: crate::inequality::InequalityError) -> Self {
        ThalesError::Inequality(e)
    }
}

impl From<crate::precision::EvalError> for ThalesError {
    fn from(e: crate::precision::EvalError) -> Self {
        ThalesError::Evaluation(e)
    }
}

impl From<crate::partial_fractions::DecomposeError> for ThalesError {
    fn from(e: crate::partial_fractions::DecomposeError) -> Self {
        ThalesError::PartialFractions(e)
    }
}

impl From<crate::latex::LaTeXParseError> for ThalesError {
    fn from(e: crate::latex::LaTeXParseError) -> Self {
        ThalesError::LaTeXParse(e)
    }
}

impl From<crate::equation_system::SystemError> for ThalesError {
    fn from(e: crate::equation_system::SystemError) -> Self {
        ThalesError::System(e)
    }
}

impl From<crate::equation_system::NonlinearSystemSolverError> for ThalesError {
    fn from(e: crate::equation_system::NonlinearSystemSolverError) -> Self {
        ThalesError::NonlinearSystem(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thales_error_from_parse_error() {
        let parse_err = crate::parser::ParseError::UnexpectedCharacter { pos: 0, found: 'x' };
        let thales_err: ThalesError = parse_err.clone().into();

        match thales_err {
            ThalesError::Parse(e) => assert_eq!(e, parse_err),
            _ => panic!("Expected ThalesError::Parse"),
        }
    }

    #[test]
    fn test_thales_error_from_solver_error() {
        let solver_err = crate::solver::SolverError::NoSolution;
        let thales_err: ThalesError = solver_err.clone().into();

        match thales_err {
            ThalesError::Solver(e) => assert_eq!(e, solver_err),
            _ => panic!("Expected ThalesError::Solver"),
        }
    }

    #[test]
    fn test_thales_error_from_numerical_error() {
        let num_err = crate::numerical::NumericalError::NoConvergence;
        let thales_err: ThalesError = num_err.clone().into();

        match thales_err {
            ThalesError::Numerical(e) => assert_eq!(e, num_err),
            _ => panic!("Expected ThalesError::Numerical"),
        }
    }

    #[test]
    fn test_thales_error_display() {
        let solver_err = crate::solver::SolverError::NoSolution;
        let thales_err: ThalesError = solver_err.into();
        let display_str = format!("{}", thales_err);

        assert!(display_str.contains("Solver error"));
        assert!(display_str.contains("NoSolution"));
    }

    #[test]
    fn test_thales_error_source() {
        use std::error::Error;

        let parse_err = crate::parser::ParseError::UnexpectedCharacter { pos: 5, found: '!' };
        let thales_err: ThalesError = parse_err.into();

        assert!(thales_err.source().is_some());
    }

    #[test]
    fn test_thales_error_from_integration_error() {
        let int_err = crate::integration::IntegrationError::DivisionByZero;
        let thales_err: ThalesError = int_err.clone().into();

        match thales_err {
            ThalesError::Integration(e) => assert_eq!(e, int_err),
            _ => panic!("Expected ThalesError::Integration"),
        }
    }

    #[test]
    fn test_thales_error_from_matrix_error() {
        let matrix_err = crate::matrix::MatrixError::EmptyMatrix;
        let thales_err: ThalesError = matrix_err.clone().into();

        match thales_err {
            ThalesError::Matrix(e) => assert_eq!(e, matrix_err),
            _ => panic!("Expected ThalesError::Matrix"),
        }
    }
}
