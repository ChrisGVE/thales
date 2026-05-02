//! Utility helpers: function-name mapping and variant-name lookup for error messages.

use crate::ast::Function;

/// Map a mathlex function name string to a thales Function enum variant.
pub(super) fn match_function_name(name: &str) -> Function {
    match name {
        // Trigonometric
        "sin" => Function::Sin,
        "cos" => Function::Cos,
        "tan" => Function::Tan,
        "arcsin" | "asin" => Function::Asin,
        "arccos" | "acos" => Function::Acos,
        "arctan" | "atan" => Function::Atan,
        "atan2" => Function::Atan2,

        // Hyperbolic
        "sinh" => Function::Sinh,
        "cosh" => Function::Cosh,
        "tanh" => Function::Tanh,

        // Exponential & Logarithmic
        "exp" => Function::Exp,
        "ln" => Function::Ln,
        "log" => Function::Log,
        "log2" | "lg" => Function::Log2,
        "log10" => Function::Log10,

        // Roots & Power
        "sqrt" => Function::Sqrt,
        "cbrt" => Function::Cbrt,
        "pow" => Function::Pow,

        // Rounding
        "floor" => Function::Floor,
        "ceil" => Function::Ceil,
        "round" => Function::Round,

        // Utility
        "abs" => Function::Abs,
        "sgn" | "sign" => Function::Sign,
        "min" => Function::Min,
        "max" => Function::Max,

        // Complex projections
        "re" | "Re" | "RE" => Function::Re,
        "im" | "Im" | "IM" => Function::Im,
        "conj" | "Conj" | "CONJ" => Function::Conj,

        // Everything else → Custom
        other => Function::Custom(other.to_string()),
    }
}

/// Get a human-readable name for a mathlex expression variant (for error messages).
pub(super) fn variant_name(expr: &mathlex::Expression) -> &'static str {
    variant_name_from_kind(&expr.kind)
}

/// Get a human-readable name for a mathlex ExprKind variant (for error messages).
pub(super) fn variant_name_from_kind(kind: &mathlex::ExprKind) -> &'static str {
    match kind {
        mathlex::ExprKind::Integer(_) => "Integer",
        mathlex::ExprKind::Float(_) => "Float",
        mathlex::ExprKind::Variable(_) => "Variable",
        mathlex::ExprKind::Constant(_) => "Constant",
        mathlex::ExprKind::Unary { .. } => "Unary",
        mathlex::ExprKind::Binary { .. } => "Binary",
        mathlex::ExprKind::Function { .. } => "Function",
        mathlex::ExprKind::Equation { .. } => "Equation",
        mathlex::ExprKind::Rational { .. } => "Rational",
        mathlex::ExprKind::Complex { .. } => "Complex",
        mathlex::ExprKind::Vector(_) => "Vector",
        mathlex::ExprKind::Matrix(_) => "Matrix",
        mathlex::ExprKind::Derivative { .. } => "Derivative",
        mathlex::ExprKind::PartialDerivative { .. } => "PartialDerivative",
        mathlex::ExprKind::Gradient { .. } => "Gradient",
        mathlex::ExprKind::Integral { .. } => "Integral",
        mathlex::ExprKind::Sum { .. } => "Sum",
        mathlex::ExprKind::Product { .. } => "Product",
        mathlex::ExprKind::Limit { .. } => "Limit",
        _ => "Unknown",
    }
}
