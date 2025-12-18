use mathsolver_core::ast::*;

// ============================================================================
// Constant Folding Tests - Binary Operations
// ============================================================================

#[test]
fn test_constant_folding_add_integers() {
    // 2 + 3 → 5
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);
    let add = Expression::Binary(BinaryOp::Add, Box::new(two), Box::new(three));

    let simplified = add.simplify();
    assert_eq!(simplified, Expression::Integer(5));
}

#[test]
fn test_constant_folding_add_floats() {
    // 2.5 + 3.7 → 6.2
    let a = Expression::Float(2.5);
    let b = Expression::Float(3.7);
    let add = Expression::Binary(BinaryOp::Add, Box::new(a), Box::new(b));

    let simplified = add.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 6.2).abs() < 1e-10);
    } else {
        panic!("Expected Float, got {:?}", simplified);
    }
}

#[test]
fn test_constant_folding_sub_integers() {
    // 10 - 3 → 7
    let ten = Expression::Integer(10);
    let three = Expression::Integer(3);
    let sub = Expression::Binary(BinaryOp::Sub, Box::new(ten), Box::new(three));

    let simplified = sub.simplify();
    assert_eq!(simplified, Expression::Integer(7));
}

#[test]
fn test_constant_folding_mul_integers() {
    // 4 * 5 → 20
    let four = Expression::Integer(4);
    let five = Expression::Integer(5);
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(four), Box::new(five));

    let simplified = mul.simplify();
    assert_eq!(simplified, Expression::Integer(20));
}

#[test]
fn test_constant_folding_div_integers() {
    // 20 / 4 → 5
    let twenty = Expression::Integer(20);
    let four = Expression::Integer(4);
    let div = Expression::Binary(BinaryOp::Div, Box::new(twenty), Box::new(four));

    let simplified = div.simplify();
    assert_eq!(simplified, Expression::Integer(5));
}

#[test]
fn test_constant_folding_div_non_integer_result() {
    // 10 / 3 → 3.333...
    let ten = Expression::Integer(10);
    let three = Expression::Integer(3);
    let div = Expression::Binary(BinaryOp::Div, Box::new(ten), Box::new(three));

    let simplified = div.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 10.0 / 3.0).abs() < 1e-10);
    } else {
        panic!("Expected Float, got {:?}", simplified);
    }
}

#[test]
fn test_constant_folding_mod() {
    // 10 % 3 → 1
    let ten = Expression::Integer(10);
    let three = Expression::Integer(3);
    let mod_op = Expression::Binary(BinaryOp::Mod, Box::new(ten), Box::new(three));

    let simplified = mod_op.simplify();
    assert_eq!(simplified, Expression::Integer(1));
}

#[test]
fn test_constant_folding_nested_operations() {
    // (2 + 3) * 4 → 5 * 4 → 20
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);
    let four = Expression::Integer(4);

    let add = Expression::Binary(BinaryOp::Add, Box::new(two), Box::new(three));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add), Box::new(four));

    let simplified = mul.simplify();
    assert_eq!(simplified, Expression::Integer(20));
}

#[test]
fn test_constant_folding_complex_nested() {
    // (10 - 2) / (1 + 1) → 8 / 2 → 4
    let ten = Expression::Integer(10);
    let two = Expression::Integer(2);
    let one1 = Expression::Integer(1);
    let one2 = Expression::Integer(1);

    let sub = Expression::Binary(BinaryOp::Sub, Box::new(ten), Box::new(two));
    let add = Expression::Binary(BinaryOp::Add, Box::new(one1), Box::new(one2));
    let div = Expression::Binary(BinaryOp::Div, Box::new(sub), Box::new(add));

    let simplified = div.simplify();
    assert_eq!(simplified, Expression::Integer(4));
}

// ============================================================================
// Function Evaluation Tests
// ============================================================================

#[test]
fn test_function_evaluation_sin_zero() {
    // sin(0) → 0
    let zero = Expression::Integer(0);
    let sin = Expression::Function(Function::Sin, vec![zero]);

    let simplified = sin.simplify();
    if let Expression::Float(result) = simplified {
        assert!(result.abs() < 1e-10);
    } else if let Expression::Integer(result) = simplified {
        assert_eq!(result, 0);
    } else {
        panic!("Expected numeric result, got {:?}", simplified);
    }
}

#[test]
fn test_function_evaluation_cos_zero() {
    // cos(0) → 1
    let zero = Expression::Integer(0);
    let cos = Expression::Function(Function::Cos, vec![zero]);

    let simplified = cos.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 1.0).abs() < 1e-10);
    } else if let Expression::Integer(result) = simplified {
        assert_eq!(result, 1);
    } else {
        panic!("Expected numeric result, got {:?}", simplified);
    }
}

#[test]
fn test_function_evaluation_sqrt() {
    // sqrt(16) → 4
    let sixteen = Expression::Integer(16);
    let sqrt = Expression::Function(Function::Sqrt, vec![sixteen]);

    let simplified = sqrt.simplify();
    assert_eq!(simplified, Expression::Integer(4));
}

#[test]
fn test_function_evaluation_sqrt_non_perfect() {
    // sqrt(2) → 1.414...
    let two = Expression::Integer(2);
    let sqrt = Expression::Function(Function::Sqrt, vec![two]);

    let simplified = sqrt.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 2.0_f64.sqrt()).abs() < 1e-10);
    } else {
        panic!("Expected Float, got {:?}", simplified);
    }
}

#[test]
fn test_function_evaluation_abs() {
    // abs(-5) → 5
    let neg_five = Expression::Integer(-5);
    let abs = Expression::Function(Function::Abs, vec![neg_five]);

    let simplified = abs.simplify();
    assert_eq!(simplified, Expression::Integer(5));
}

#[test]
fn test_function_evaluation_ln_e() {
    // ln(e) → 1
    let e = Expression::Float(std::f64::consts::E);
    let ln = Expression::Function(Function::Ln, vec![e]);

    let simplified = ln.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 1.0).abs() < 1e-10);
    } else if let Expression::Integer(result) = simplified {
        assert_eq!(result, 1);
    } else {
        panic!("Expected numeric result, got {:?}", simplified);
    }
}

#[test]
fn test_function_evaluation_exp_zero() {
    // exp(0) → 1
    let zero = Expression::Integer(0);
    let exp = Expression::Function(Function::Exp, vec![zero]);

    let simplified = exp.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 1.0).abs() < 1e-10);
    } else if let Expression::Integer(result) = simplified {
        assert_eq!(result, 1);
    } else {
        panic!("Expected numeric result, got {:?}", simplified);
    }
}

#[test]
fn test_function_evaluation_log10() {
    // log10(100) → 2
    let hundred = Expression::Integer(100);
    let log10 = Expression::Function(Function::Log10, vec![hundred]);

    let simplified = log10.simplify();
    assert_eq!(simplified, Expression::Integer(2));
}

#[test]
fn test_function_evaluation_max() {
    // max(5, 3) → 5
    let five = Expression::Integer(5);
    let three = Expression::Integer(3);
    let max = Expression::Function(Function::Max, vec![five, three]);

    let simplified = max.simplify();
    assert_eq!(simplified, Expression::Integer(5));
}

#[test]
fn test_function_evaluation_min() {
    // min(5, 3) → 3
    let five = Expression::Integer(5);
    let three = Expression::Integer(3);
    let min = Expression::Function(Function::Min, vec![five, three]);

    let simplified = min.simplify();
    assert_eq!(simplified, Expression::Integer(3));
}

#[test]
fn test_function_evaluation_floor() {
    // floor(3.7) → 3
    let val = Expression::Float(3.7);
    let floor = Expression::Function(Function::Floor, vec![val]);

    let simplified = floor.simplify();
    assert_eq!(simplified, Expression::Integer(3));
}

#[test]
fn test_function_evaluation_ceil() {
    // ceil(3.2) → 4
    let val = Expression::Float(3.2);
    let ceil = Expression::Function(Function::Ceil, vec![val]);

    let simplified = ceil.simplify();
    assert_eq!(simplified, Expression::Integer(4));
}

#[test]
fn test_function_evaluation_round() {
    // round(3.6) → 4
    let val = Expression::Float(3.6);
    let round = Expression::Function(Function::Round, vec![val]);

    let simplified = round.simplify();
    assert_eq!(simplified, Expression::Integer(4));
}

// ============================================================================
// Power Constant Folding Tests
// ============================================================================

#[test]
fn test_power_constant_folding() {
    // 2^3 → 8
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);
    let pow = Expression::Power(Box::new(two), Box::new(three));

    let simplified = pow.simplify();
    assert_eq!(simplified, Expression::Integer(8));
}

#[test]
fn test_power_constant_folding_float() {
    // 2^0.5 → 1.414...
    let two = Expression::Integer(2);
    let half = Expression::Float(0.5);
    let pow = Expression::Power(Box::new(two), Box::new(half));

    let simplified = pow.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 2.0_f64.sqrt()).abs() < 1e-10);
    } else {
        panic!("Expected Float, got {:?}", simplified);
    }
}

#[test]
fn test_power_constant_folding_negative_exponent() {
    // 2^(-2) → 0.25
    let two = Expression::Integer(2);
    let neg_two = Expression::Integer(-2);
    let pow = Expression::Power(Box::new(two), Box::new(neg_two));

    let simplified = pow.simplify();
    if let Expression::Float(result) = simplified {
        assert!((result - 0.25).abs() < 1e-10);
    } else {
        panic!("Expected Float, got {:?}", simplified);
    }
}

// ============================================================================
// Mixed Tests - Constants and Variables
// ============================================================================

#[test]
fn test_constant_folding_with_variable() {
    // (2 + 3) + x → 5 + x
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);
    let x = Expression::Variable(Variable::new("x"));

    let add1 = Expression::Binary(BinaryOp::Add, Box::new(two), Box::new(three));
    let add2 = Expression::Binary(BinaryOp::Add, Box::new(add1), Box::new(x.clone()));

    let simplified = add2.simplify();

    // Should be 5 + x
    if let Expression::Binary(BinaryOp::Add, left, right) = simplified {
        assert_eq!(*left, Expression::Integer(5));
        assert_eq!(*right, x);
    } else {
        panic!("Expected Binary Add, got {:?}", simplified);
    }
}

#[test]
fn test_constant_folding_preserves_variables() {
    // x * (2 * 3) → x * 6
    let x = Expression::Variable(Variable::new("x"));
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);

    let mul1 = Expression::Binary(BinaryOp::Mul, Box::new(two), Box::new(three));
    let mul2 = Expression::Binary(BinaryOp::Mul, Box::new(x.clone()), Box::new(mul1));

    let simplified = mul2.simplify();

    // Should be x * 6
    if let Expression::Binary(BinaryOp::Mul, left, right) = simplified {
        assert_eq!(*left, x);
        assert_eq!(*right, Expression::Integer(6));
    } else {
        panic!("Expected Binary Mul, got {:?}", simplified);
    }
}

#[test]
fn test_function_not_evaluated_with_variable() {
    // sin(x) → sin(x) (no simplification)
    let x = Expression::Variable(Variable::new("x"));
    let sin = Expression::Function(Function::Sin, vec![x.clone()]);

    let simplified = sin.simplify();

    // Should remain as sin(x)
    if let Expression::Function(Function::Sin, args) = simplified {
        assert_eq!(args.len(), 1);
        assert_eq!(args[0], x);
    } else {
        panic!("Expected Function Sin, got {:?}", simplified);
    }
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

#[test]
fn test_complex_nested_simplification() {
    // ((x + 0) * 1) + (2 + 3) → x + 5
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let one = Expression::Integer(1);
    let two = Expression::Integer(2);
    let three = Expression::Integer(3);

    let add1 = Expression::Binary(BinaryOp::Add, Box::new(x.clone()), Box::new(zero));
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(add1), Box::new(one));
    let add2 = Expression::Binary(BinaryOp::Add, Box::new(two), Box::new(three));
    let final_add = Expression::Binary(BinaryOp::Add, Box::new(mul), Box::new(add2));

    let simplified = final_add.simplify();

    // Should be x + 5
    if let Expression::Binary(BinaryOp::Add, left, right) = simplified {
        assert_eq!(*left, x);
        assert_eq!(*right, Expression::Integer(5));
    } else {
        panic!("Expected Binary Add, got {:?}", simplified);
    }
}

#[test]
fn test_function_with_constant_argument_from_simplification() {
    // sqrt(2 + 2) → sqrt(4) → 2
    let two1 = Expression::Integer(2);
    let two2 = Expression::Integer(2);
    let add = Expression::Binary(BinaryOp::Add, Box::new(two1), Box::new(two2));
    let sqrt = Expression::Function(Function::Sqrt, vec![add]);

    let simplified = sqrt.simplify();
    assert_eq!(simplified, Expression::Integer(2));
}

#[test]
fn test_power_with_constant_from_simplification() {
    // (1 + 1)^(2 + 1) → 2^3 → 8
    let one1 = Expression::Integer(1);
    let one2 = Expression::Integer(1);
    let two = Expression::Integer(2);
    let one3 = Expression::Integer(1);

    let base = Expression::Binary(BinaryOp::Add, Box::new(one1), Box::new(one2));
    let exp = Expression::Binary(BinaryOp::Add, Box::new(two), Box::new(one3));
    let pow = Expression::Power(Box::new(base), Box::new(exp));

    let simplified = pow.simplify();
    assert_eq!(simplified, Expression::Integer(8));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_division_by_zero_not_folded() {
    // 5 / 0 → should not fold (avoid division by zero)
    let five = Expression::Integer(5);
    let zero = Expression::Integer(0);
    let div = Expression::Binary(BinaryOp::Div, Box::new(five.clone()), Box::new(zero.clone()));

    let simplified = div.simplify();

    // Should remain as 5 / 0 (not folded)
    if let Expression::Binary(BinaryOp::Div, left, right) = simplified {
        assert_eq!(*left, five);
        assert_eq!(*right, zero);
    } else {
        panic!("Expected Binary Div, got {:?}", simplified);
    }
}

#[test]
fn test_very_small_result_becomes_zero() {
    // sin(very_small) should be close to 0
    let very_small = Expression::Float(1e-20);
    let sin = Expression::Function(Function::Sin, vec![very_small]);

    let simplified = sin.simplify();
    if let Expression::Float(result) = simplified {
        assert!(result.abs() < 1e-10);
    } else if let Expression::Integer(result) = simplified {
        assert_eq!(result, 0);
    } else {
        panic!("Expected numeric result, got {:?}", simplified);
    }
}

#[test]
fn test_triple_negation() {
    // -(-(-x)) → -x
    let x = Expression::Variable(Variable::new("x"));
    let neg1 = Expression::Unary(UnaryOp::Neg, Box::new(x.clone()));
    let neg2 = Expression::Unary(UnaryOp::Neg, Box::new(neg1));
    let neg3 = Expression::Unary(UnaryOp::Neg, Box::new(neg2));

    let simplified = neg3.simplify();

    // Should be -x
    if let Expression::Unary(UnaryOp::Neg, inner) = simplified {
        assert_eq!(*inner, x);
    } else {
        panic!("Expected Unary Neg, got {:?}", simplified);
    }
}

#[test]
fn test_identity_operations_priority_over_folding() {
    // x * 0 → 0 (identity takes priority over any other simplification)
    let x = Expression::Variable(Variable::new("x"));
    let zero = Expression::Integer(0);
    let mul = Expression::Binary(BinaryOp::Mul, Box::new(x), Box::new(zero));

    let simplified = mul.simplify();
    assert_eq!(simplified, Expression::Integer(0));
}
