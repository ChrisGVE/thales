//! Unit tests for the mathematical expression parser.

use mathsolver_core::ast::{BinaryOp, Expression, Function, UnaryOp};
use mathsolver_core::parser::{parse_equation, parse_expression};

#[test]
fn test_simple_integer() {
    let result = parse_expression("42");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert_eq!(n, 42.0),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_simple_addition() {
    let result = parse_expression("2 + 3");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Add, left, right) => match (*left, *right) {
            (Expression::Float(l), Expression::Float(r)) => {
                assert_eq!(l, 2.0);
                assert_eq!(r, 3.0);
            }
            _ => panic!("Expected Float operands"),
        },
        _ => panic!("Expected Binary Add"),
    }
}

#[test]
fn test_simple_multiplication() {
    let result = parse_expression("x * y");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Mul, left, right) => match (*left, *right) {
            (Expression::Variable(v1), Expression::Variable(v2)) => {
                assert_eq!(v1.name, "x");
                assert_eq!(v2.name, "y");
            }
            _ => panic!("Expected Variable operands"),
        },
        _ => panic!("Expected Binary Mul"),
    }
}

#[test]
fn test_power_operation() {
    let result = parse_expression("a ^ 2");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Power(base, exp) => match (*base, *exp) {
            (Expression::Variable(v), Expression::Float(n)) => {
                assert_eq!(v.name, "a");
                assert_eq!(n, 2.0);
            }
            _ => panic!("Expected Variable base and Float exponent"),
        },
        _ => panic!("Expected Power"),
    }
}

#[test]
fn test_negation() {
    let result = parse_expression("-x");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Unary(UnaryOp::Neg, expr) => match *expr {
            Expression::Variable(v) => assert_eq!(v.name, "x"),
            _ => panic!("Expected Variable"),
        },
        _ => panic!("Expected Unary Neg"),
    }
}

#[test]
fn test_operator_precedence_mul_add() {
    let result = parse_expression("2 + 3 * 4");
    assert!(result.is_ok());
    // Should parse as 2 + (3 * 4)
    match result.unwrap() {
        Expression::Binary(BinaryOp::Add, left, right) => {
            match *left {
                Expression::Float(n) => assert_eq!(n, 2.0),
                _ => panic!("Expected Float on left"),
            }
            match *right {
                Expression::Binary(BinaryOp::Mul, l, r) => match (*l, *r) {
                    (Expression::Float(n1), Expression::Float(n2)) => {
                        assert_eq!(n1, 3.0);
                        assert_eq!(n2, 4.0);
                    }
                    _ => panic!("Expected Float operands in multiplication"),
                },
                _ => panic!("Expected Binary Mul on right"),
            }
        }
        _ => panic!("Expected Binary Add at top level"),
    }
}

#[test]
fn test_operator_precedence_power_mul() {
    let result = parse_expression("2 * 3 ^ 4");
    assert!(result.is_ok());
    // Should parse as 2 * (3 ^ 4)
    match result.unwrap() {
        Expression::Binary(BinaryOp::Mul, left, right) => {
            match *left {
                Expression::Float(n) => assert_eq!(n, 2.0),
                _ => panic!("Expected Float on left"),
            }
            match *right {
                Expression::Power(base, exp) => match (*base, *exp) {
                    (Expression::Float(b), Expression::Float(e)) => {
                        assert_eq!(b, 3.0);
                        assert_eq!(e, 4.0);
                    }
                    _ => panic!("Expected Float operands in power"),
                },
                _ => panic!("Expected Power on right"),
            }
        }
        _ => panic!("Expected Binary Mul at top level"),
    }
}

#[test]
fn test_right_associative_power() {
    let result = parse_expression("2 ^ 3 ^ 4");
    assert!(result.is_ok());
    // Should parse as 2 ^ (3 ^ 4), not (2 ^ 3) ^ 4
    match result.unwrap() {
        Expression::Power(base, exp) => {
            match *base {
                Expression::Float(n) => assert_eq!(n, 2.0),
                _ => panic!("Expected Float base"),
            }
            match *exp {
                Expression::Power(inner_base, inner_exp) => match (*inner_base, *inner_exp) {
                    (Expression::Float(b), Expression::Float(e)) => {
                        assert_eq!(b, 3.0);
                        assert_eq!(e, 4.0);
                    }
                    _ => panic!("Expected Float operands in inner power"),
                },
                _ => panic!("Expected Power as exponent"),
            }
        }
        _ => panic!("Expected Power at top level"),
    }
}

#[test]
fn test_function_sin() {
    let result = parse_expression("sin(x)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Function(func, args) => {
            assert_eq!(func, Function::Sin);
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Variable(v) => assert_eq!(v.name, "x"),
                _ => panic!("Expected Variable argument"),
            }
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_function_log_two_args() {
    let result = parse_expression("log(10, x)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Function(func, args) => {
            assert_eq!(func, Function::Log);
            assert_eq!(args.len(), 2);
            match (&args[0], &args[1]) {
                (Expression::Float(n), Expression::Variable(v)) => {
                    assert_eq!(*n, 10.0);
                    assert_eq!(v.name, "x");
                }
                _ => panic!("Expected Float and Variable arguments"),
            }
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_function_sqrt_nested() {
    let result = parse_expression("sqrt(a^2 + b^2)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Function(func, args) => {
            assert_eq!(func, Function::Sqrt);
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Binary(BinaryOp::Add, _, _) => (),
                _ => panic!("Expected Binary Add inside sqrt"),
            }
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_equation_simple() {
    let result = parse_equation("x + 2 = 5");
    assert!(result.is_ok());
    let eq = result.unwrap();
    match eq.left {
        Expression::Binary(BinaryOp::Add, _, _) => (),
        _ => panic!("Expected Binary Add on left"),
    }
    match eq.right {
        Expression::Float(n) => assert_eq!(n, 5.0),
        _ => panic!("Expected Float on right"),
    }
}

#[test]
fn test_equation_f_equals_ma() {
    let result = parse_equation("F = m * a");
    assert!(result.is_ok());
    let eq = result.unwrap();
    match eq.left {
        Expression::Variable(v) => assert_eq!(v.name, "F"),
        _ => panic!("Expected Variable on left"),
    }
    match eq.right {
        Expression::Binary(BinaryOp::Mul, _, _) => (),
        _ => panic!("Expected Binary Mul on right"),
    }
}

#[test]
fn test_equation_e_equals_mc2() {
    let result = parse_equation("E = m * c ^ 2");
    assert!(result.is_ok());
    let eq = result.unwrap();
    match eq.left {
        Expression::Variable(v) => assert_eq!(v.name, "E"),
        _ => panic!("Expected Variable on left"),
    }
    match eq.right {
        Expression::Binary(BinaryOp::Mul, left, right) => match (*left, *right) {
            (Expression::Variable(m), Expression::Power(_, _)) => {
                assert_eq!(m.name, "m");
            }
            _ => panic!("Expected Variable and Power"),
        },
        _ => panic!("Expected Binary Mul on right"),
    }
}

#[test]
fn test_scientific_notation_positive_exp() {
    let result = parse_expression("1.5e3");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert_eq!(n, 1500.0),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_scientific_notation_negative_exp() {
    let result = parse_expression("1.5e-3");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert!((n - 0.0015).abs() < 1e-10),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_scientific_notation_large() {
    let result = parse_expression("6.02e23");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert_eq!(n, 6.02e23),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_variable_x() {
    let result = parse_expression("x");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Variable(v) => assert_eq!(v.name, "x"),
        _ => panic!("Expected Variable"),
    }
}

#[test]
fn test_variable_theta() {
    let result = parse_expression("theta");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Variable(v) => assert_eq!(v.name, "theta"),
        _ => panic!("Expected Variable"),
    }
}

#[test]
fn test_variable_x1() {
    let result = parse_expression("x1");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Variable(v) => assert_eq!(v.name, "x1"),
        _ => panic!("Expected Variable"),
    }
}

#[test]
fn test_complex_expression_with_parens() {
    let result = parse_expression("(a + b) * (c - d)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Mul, left, right) => match (*left, *right) {
            (Expression::Binary(BinaryOp::Add, _, _), Expression::Binary(BinaryOp::Sub, _, _)) => {
                ()
            }
            _ => panic!("Expected Add and Sub inside Mul"),
        },
        _ => panic!("Expected Binary Mul"),
    }
}

#[test]
fn test_pythagorean_identity() {
    let result = parse_expression("sin(x)^2 + cos(x)^2");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Add, left, right) => match (*left, *right) {
            (Expression::Power(l_base, _), Expression::Power(r_base, _)) => {
                match (*l_base, *r_base) {
                    (
                        Expression::Function(Function::Sin, _),
                        Expression::Function(Function::Cos, _),
                    ) => (),
                    _ => panic!("Expected sin and cos functions"),
                }
            }
            _ => panic!("Expected Power operands"),
        },
        _ => panic!("Expected Binary Add"),
    }
}

#[test]
fn test_nested_parentheses() {
    let result = parse_expression("((x + 1) * 2) - 3");
    assert!(result.is_ok());
    // Just verify it parses successfully and has correct top-level structure
    match result.unwrap() {
        Expression::Binary(BinaryOp::Sub, _, _) => (),
        _ => panic!("Expected Binary Sub at top level"),
    }
}

#[test]
fn test_multiple_operations() {
    let result = parse_expression("a + b * c - d / e");
    assert!(result.is_ok());
    // Verify it parses - detailed structure would be: (a + (b * c)) - (d / e)
    match result.unwrap() {
        Expression::Binary(BinaryOp::Sub, left, right) => match (*left, *right) {
            (Expression::Binary(BinaryOp::Add, _, _), Expression::Binary(BinaryOp::Div, _, _)) => {
                ()
            }
            _ => panic!("Expected Add on left, Div on right"),
        },
        _ => panic!("Expected Binary Sub at top level"),
    }
}

#[test]
fn test_float_number() {
    let result = parse_expression("3.14159");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert!((n - 3.14159).abs() < 1e-10),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_multiple_functions() {
    let result = parse_expression("sin(x) + cos(y) - tan(z)");
    assert!(result.is_ok());
    // Just verify it parses successfully
    match result.unwrap() {
        Expression::Binary(BinaryOp::Sub, _, _) => (),
        _ => panic!("Expected Binary Sub at top level"),
    }
}

#[test]
fn test_error_unclosed_paren() {
    let result = parse_expression("(2 + 3");
    assert!(result.is_err());
}

#[test]
fn test_error_invalid_token() {
    let result = parse_expression("2 + @ 3");
    assert!(result.is_err());
}

#[test]
fn test_error_unknown_function() {
    let result = parse_expression("unknown_func(x)");
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(!errors.is_empty());
    // Check that error message mentions unknown function
    let error_str = format!("{:?}", errors[0]);
    assert!(error_str.contains("Unknown function") || error_str.contains("unknown_func"));
}

#[test]
fn test_error_empty_input() {
    let result = parse_expression("");
    assert!(result.is_err());
}

#[test]
fn test_division_operator() {
    let result = parse_expression("a / b");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Div, _, _) => (),
        _ => panic!("Expected Binary Div"),
    }
}

#[test]
fn test_modulo_operator() {
    let result = parse_expression("a % b");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Mod, _, _) => (),
        _ => panic!("Expected Binary Mod"),
    }
}

#[test]
fn test_double_negation() {
    let result = parse_expression("--x");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Unary(UnaryOp::Neg, inner) => match *inner {
            Expression::Unary(UnaryOp::Neg, _) => (),
            _ => panic!("Expected nested Neg"),
        },
        _ => panic!("Expected Unary Neg"),
    }
}

#[test]
fn test_whitespace_handling() {
    let result = parse_expression("  2   +   3  ");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Add, _, _) => (),
        _ => panic!("Expected Binary Add"),
    }
}

#[test]
fn test_no_spaces() {
    let result = parse_expression("2+3*4");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Binary(BinaryOp::Add, _, _) => (),
        _ => panic!("Expected Binary Add"),
    }
}

#[test]
fn test_function_max() {
    let result = parse_expression("max(a, b, c)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Function(func, args) => {
            assert_eq!(func, Function::Max);
            assert_eq!(args.len(), 3);
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_function_min() {
    let result = parse_expression("min(x, y)");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Function(func, args) => {
            assert_eq!(func, Function::Min);
            assert_eq!(args.len(), 2);
        }
        _ => panic!("Expected Function"),
    }
}

#[test]
fn test_all_trig_functions() {
    let funcs = vec![
        ("sin(x)", Function::Sin),
        ("cos(x)", Function::Cos),
        ("tan(x)", Function::Tan),
        ("asin(x)", Function::Asin),
        ("acos(x)", Function::Acos),
        ("atan(x)", Function::Atan),
    ];

    for (input, expected_func) in funcs {
        let result = parse_expression(input);
        assert!(result.is_ok(), "Failed to parse: {}", input);
        match result.unwrap() {
            Expression::Function(func, _) => assert_eq!(func, expected_func),
            _ => panic!("Expected Function for {}", input),
        }
    }
}

#[test]
fn test_hyperbolic_functions() {
    let funcs = vec![
        ("sinh(x)", Function::Sinh),
        ("cosh(x)", Function::Cosh),
        ("tanh(x)", Function::Tanh),
    ];

    for (input, expected_func) in funcs {
        let result = parse_expression(input);
        assert!(result.is_ok(), "Failed to parse: {}", input);
        match result.unwrap() {
            Expression::Function(func, _) => assert_eq!(func, expected_func),
            _ => panic!("Expected Function for {}", input),
        }
    }
}

#[test]
fn test_logarithmic_functions() {
    let funcs = vec![
        ("ln(x)", Function::Ln),
        ("log10(x)", Function::Log10),
        ("log2(x)", Function::Log2),
        ("exp(x)", Function::Exp),
    ];

    for (input, expected_func) in funcs {
        let result = parse_expression(input);
        assert!(result.is_ok(), "Failed to parse: {}", input);
        match result.unwrap() {
            Expression::Function(func, _) => assert_eq!(func, expected_func),
            _ => panic!("Expected Function for {}", input),
        }
    }
}

#[test]
fn test_rounding_functions() {
    let funcs = vec![
        ("floor(x)", Function::Floor),
        ("ceil(x)", Function::Ceil),
        ("round(x)", Function::Round),
    ];

    for (input, expected_func) in funcs {
        let result = parse_expression(input);
        assert!(result.is_ok(), "Failed to parse: {}", input);
        match result.unwrap() {
            Expression::Function(func, _) => assert_eq!(func, expected_func),
            _ => panic!("Expected Function for {}", input),
        }
    }
}

#[test]
fn test_integer_only() {
    let result = parse_expression("42");
    assert!(result.is_ok());
}

#[test]
fn test_float_with_decimal() {
    let result = parse_expression("3.14");
    assert!(result.is_ok());
}

#[test]
fn test_scientific_e_upper() {
    let result = parse_expression("1E6");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert_eq!(n, 1e6),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_scientific_e_lower() {
    let result = parse_expression("1e6");
    assert!(result.is_ok());
    match result.unwrap() {
        Expression::Float(n) => assert_eq!(n, 1e6),
        _ => panic!("Expected Float"),
    }
}
