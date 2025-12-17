use mathsolver_core::ast::*;
use std::collections::HashMap;

#[test]
fn test_ast_construction_integer() {
    let expr = Expression::Integer(42);
    assert_eq!(format!("{}", expr), "42");
}

#[test]
fn test_ast_construction_float() {
    let expr = Expression::Float(3.14);
    assert_eq!(format!("{}", expr), "3.14");
}

#[test]
fn test_ast_construction_variable() {
    let var = Variable::new("x");
    let expr = Expression::Variable(var);
    assert_eq!(format!("{}", expr), "x");
}

#[test]
fn test_ast_construction_binary_add() {
    let left = Box::new(Expression::Variable(Variable::new("x")));
    let right = Box::new(Expression::Integer(5));
    let expr = Expression::Binary(BinaryOp::Add, left, right);
    assert_eq!(format!("{}", expr), "x + 5");
}

#[test]
fn test_ast_construction_complex_expression() {
    // (x + 2) * 3
    let x = Expression::Variable(Variable::new("x"));
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);

    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(two));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add), Box::new(three));

    assert_eq!(format!("{}", mul), "(x + 2) * 3");
}

#[test]
fn test_display_operator_precedence_add_mul() {
    // x + y * z should display as "x + y * z" without extra parentheses
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let z = Expression::Variable(Variable::new("z"));

    let mul = Expression::Binary(BinaryOp::Mul, Box::new(y), Box::new(z));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(mul));

    assert_eq!(format!("{}", add), "x + y * z");
}

#[test]
fn test_display_operator_precedence_mul_add() {
    // (x + y) * z should display with parentheses
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let z = Expression::Variable(Variable::new("z"));

    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add), Box::new(z));

    assert_eq!(format!("{}", mul), "(x + y) * z");
}

#[test]
fn test_display_negation() {
    let x = Expression::Variable(Variable::new("x"));
    let neg = Expression::Unary(UnaryOp::Neg, Box::new(x));

    assert_eq!(format!("{}", neg), "-x");
}

#[test]
fn test_display_absolute_value() {
    let x = Expression::Variable(Variable::new("x"));
    let abs = Expression::Unary(UnaryOp::Abs, Box::new(x));

    assert_eq!(format!("{}", abs), "|x|");
}

#[test]
fn test_display_function_call() {
    let x = Expression::Variable(Variable::new("x"));
    let sin = Expression::Function(Function::Sin, vec![x]);

    assert_eq!(format!("{}", sin), "sin(x)");
}

#[test]
fn test_display_function_call_multiple_args() {
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let max = Expression::Function(Function::Max, vec![x, y]);

    assert_eq!(format!("{}", max), "max(x, y)");
}

#[test]
fn test_display_power() {
    let x = Expression::Variable(Variable::new("x"));
    let two = Expression::Integer(2);
    let pow = Expression::Power(Box::new(x), Box::new(two));

    assert_eq!(format!("{}", pow), "x^2");
}

#[test]
fn test_variable_extraction_single() {
    let x = Expression::Variable(Variable::new("x"));
    let vars = x.variables();

    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_variable_extraction_multiple() {
    // x + y * z
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let z = Expression::Variable(Variable::new("z"));

    let mul = Expression::Binary(BinaryOp::Mul, Box::new(y), Box::new(z));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(mul));

    let vars = add.variables();

    assert_eq!(vars.len(), 3);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
    assert!(vars.contains("z"));
}

#[test]
fn test_variable_extraction_duplicate() {
    // x + x
    let x1 = Expression::Variable(Variable::new("x"));
    let x2 = Expression::Variable(Variable::new("x"));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x1), Box::new(x2));

    let vars = add.variables();

    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_variable_extraction_none() {
    let expr = Expression::Integer(42);
    let vars = expr.variables();

    assert_eq!(vars.len(), 0);
}

#[test]
fn test_contains_variable_true() {
    let x = Expression::Variable(Variable::new("x"));
    assert!(x.contains_variable("x"));
}

#[test]
fn test_contains_variable_false() {
    let x = Expression::Variable(Variable::new("x"));
    assert!(!x.contains_variable("y"));
}

#[test]
fn test_contains_variable_in_expression() {
    // x + y
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y));

    assert!(add.contains_variable("x"));
    assert!(add.contains_variable("y"));
    assert!(!add.contains_variable("z"));
}

#[test]
fn test_simplify_add_zero_left() {
    // 0 + x → x
    let zero = Expression::Integer(0);
    let x = Expression::Variable(Variable::new("x"));
    let add = Expression::Binary(BinaryOp::Add, Box::new(zero), Box::new(x.clone()));

    let simplified = add.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_add_zero_right() {
    // x + 0 → x
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let add = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(zero));

    let simplified = add.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_sub_zero() {
    // x - 0 → x
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let sub = Expression::Binary(BinaryOp::Sub, Box::new(x.clone()), Box::new(zero));

    let simplified = sub.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_mul_zero_left() {
    // 0 * x → 0
    let zero = Expression::Integer(0);
    let x = Expression::Variable(Variable::new("x"));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(zero), Box::new(x));

    let simplified = mul.simplify();
    assert_eq!(simplified, Expression::Integer(0));
}

#[test]
fn test_simplify_mul_zero_right() {
    // x * 0 → 0
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(zero));

    let simplified = mul.simplify();
    assert_eq!(simplified, Expression::Integer(0));
}

#[test]
fn test_simplify_mul_one_left() {
    // 1 * x → x
    let one = Expression::Integer(1);
    let x = Expression::Variable(Variable::new("x"));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(one), Box::new(x.clone()));

    let simplified = mul.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_mul_one_right() {
    // x * 1 → x
    let x = Expression::Variable(Variable::new("x"));
    let one = Expression::Integer(1);
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(x.clone()), Box::new(one));

    let simplified = mul.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_div_one() {
    // x / 1 → x
    let x = Expression::Variable(Variable::new("x"));
    let one = Expression::Integer(1);
    let div = Expression::Binary(BinaryOp::Div, Box::new(x.clone()), Box::new(one));

    let simplified = div.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_power_zero() {
    // x^0 → 1
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let pow = Expression::Power(Box::new(x), Box::new(zero));

    let simplified = pow.simplify();
    assert_eq!(simplified, Expression::Integer(1));
}

#[test]
fn test_simplify_power_one() {
    // x^1 → x
    let x = Expression::Variable(Variable::new("x"));
    let one = Expression::Integer(1);
    let pow = Expression::Power(Box::new(x.clone()), Box::new(one));

    let simplified = pow.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_double_negation() {
    // -(-x) → x
    let x = Expression::Variable(Variable::new("x"));
    let neg1 = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
    let neg2 = Expression::Unary(UnaryOp::Neg, Box::new(neg1));

    let simplified = neg2.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_nested() {
    // (x + 0) * 1 → x
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let one = Expression::Integer(1);

    let add = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(zero));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add), Box::new(one));

    let simplified = mul.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_evaluate_integer() {
    let expr = Expression::Integer(42);
    let vars = HashMap::new();

    assert_eq!(expr.evaluate(&vars), Some(42.0));
}

#[test]
fn test_evaluate_float() {
    let expr = Expression::Float(3.14);
    let vars = HashMap::new();

    assert_eq!(expr.evaluate(&vars), Some(3.14));
}

#[test]
fn test_evaluate_variable_present() {
    let x = Expression::Variable(Variable::new("x"));
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(x.evaluate(&vars), Some(10.0));
}

#[test]
fn test_evaluate_variable_missing() {
    let x = Expression::Variable(Variable::new("x"));
    let vars = HashMap::new();

    assert_eq!(x.evaluate(&vars), None);
}

#[test]
fn test_evaluate_add() {
    // x + 5
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(five));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(add.evaluate(&vars), Some(15.0));
}

#[test]
fn test_evaluate_sub() {
    // x - 5
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);
    let sub = Expression::Binary(BinaryOp::Sub, Box::new(x), Box::new(five));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(sub.evaluate(&vars), Some(5.0));
}

#[test]
fn test_evaluate_mul() {
    // x * 5
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(five));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(mul.evaluate(&vars), Some(50.0));
}

#[test]
fn test_evaluate_div() {
    // x / 5
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);
    let div = Expression::Binary(BinaryOp::Div, Box::new(x), Box::new(five));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(div.evaluate(&vars), Some(2.0));
}

#[test]
fn test_evaluate_div_by_zero() {
    // x / 0
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let div = Expression::Binary(BinaryOp::Div, Box::new(x), Box::new(zero));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(div.evaluate(&vars), None);
}

#[test]
fn test_evaluate_negation() {
    // -x
    let x = Expression::Variable(Variable::new("x"));
    let neg = Expression::Unary(UnaryOp::Neg, Box::new(x));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(neg.evaluate(&vars), Some(-10.0));
}

#[test]
fn test_evaluate_abs() {
    // |x|
    let x = Expression::Variable(Variable::new("x"));
    let abs = Expression::Unary(UnaryOp::Abs, Box::new(x));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), -10.0);

    assert_eq!(abs.evaluate(&vars), Some(10.0));
}

#[test]
fn test_evaluate_power() {
    // x^2
    let x = Expression::Variable(Variable::new("x"));
    let two = Expression::Integer(2);
    let pow = Expression::Power(Box::new(x), Box::new(two));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    assert_eq!(pow.evaluate(&vars), Some(9.0));
}

#[test]
fn test_evaluate_sin() {
    use std::f64::consts::PI;

    // sin(x)
    let x = Expression::Variable(Variable::new("x"));
    let sin = Expression::Function(Function::Sin, vec![x]);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), PI / 2.0);

    let result = sin.evaluate(&vars).unwrap();
    assert!((result - 1.0).abs() < 1e-10);
}

#[test]
fn test_evaluate_sqrt() {
    // sqrt(x)
    let x = Expression::Variable(Variable::new("x"));
    let sqrt = Expression::Function(Function::Sqrt, vec![x]);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 16.0);

    assert_eq!(sqrt.evaluate(&vars), Some(4.0));
}

#[test]
fn test_evaluate_complex_expression() {
    // (x + 2) * y
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let two = Expression::Integer(2);

    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(two));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add), Box::new(y));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 4.0);

    assert_eq!(mul.evaluate(&vars), Some(20.0));
}

#[test]
fn test_evaluate_missing_variable() {
    // x + y, but only x is provided
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(y));

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 10.0);

    assert_eq!(add.evaluate(&vars), None);
}

#[test]
fn test_map_double_values() {
    // x + 5, map to double all integers
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(five));

    let doubled = add.map(&|expr| {
        if let Expression::Integer(n) = expr {
            Expression::Integer(n * 2)
        } else {
            expr.clone()
        }
    });

    // Result should be x + 10
    let expected_x = Expression::Variable(Variable::new("x"));
    let expected_ten = Expression::Integer(10);
    let expected = Expression::Binary(BinaryOp::Add, Box::new(expected_x), Box::new(expected_ten));

    assert_eq!(doubled, expected);
}

#[test]
fn test_fold_count_nodes() {
    // x + y * z
    let x = Expression::Variable(Variable::new("x"));
    let y = Expression::Variable(Variable::new("y"));
    let z = Expression::Variable(Variable::new("z"));

    let mul = Expression::Binary(BinaryOp::Mul, Box::new(y), Box::new(z));
    let add = Expression::Binary(BinaryOp::Add, Box::new(x), Box::new(mul));

    let count = add.fold(0, &|acc, _expr| acc + 1);

    // Should count: add, x, mul, y, z = 5 nodes
    assert_eq!(count, 5);
}

#[test]
fn test_equation_creation() {
    let x = Expression::Variable(Variable::new("x"));
    let five = Expression::Integer(5);

    let eq = Equation::new("eq1", x, five);

    assert_eq!(eq.id, "eq1");
    assert_eq!(format!("{}", eq.left), "x");
    assert_eq!(format!("{}", eq.right), "5");
}
