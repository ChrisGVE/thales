//! Performance benchmarks for mathsolver-core.
//!
//! Benchmarks critical operations using Criterion.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathsolver_core::transforms::{Cartesian2D, Cartesian3D, Polar, Spherical};

/// Benchmark coordinate transformations.
fn bench_coordinate_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_transforms");

    // 2D transformations
    group.bench_function("cartesian_to_polar", |b| {
        let cart = Cartesian2D::new(3.0, 4.0);
        b.iter(|| black_box(cart.to_polar()));
    });

    group.bench_function("polar_to_cartesian", |b| {
        let polar = Polar::new(5.0, 0.927295218);
        b.iter(|| black_box(polar.to_cartesian()));
    });

    // 3D transformations
    group.bench_function("cartesian_to_spherical", |b| {
        let cart = Cartesian3D::new(1.0, 1.0, 1.0);
        b.iter(|| black_box(cart.to_spherical()));
    });

    group.bench_function("spherical_to_cartesian", |b| {
        let spherical = Spherical::new(1.732050808, 0.7853981634, 0.9553166181);
        b.iter(|| black_box(spherical.to_cartesian()));
    });

    group.finish();
}

/// Benchmark parsing operations.
fn bench_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");

    // TODO: Add parsing benchmarks when parser is implemented
    // group.bench_function("parse_simple_equation", |b| {
    //     b.iter(|| black_box(mathsolver_core::parse_equation("x + 2 = 5")));
    // });

    // group.bench_function("parse_complex_expression", |b| {
    //     b.iter(|| black_box(mathsolver_core::parse_expression("sin(x) + cos(y) * tan(z)")));
    // });

    // group.bench_function("parse_polynomial", |b| {
    //     b.iter(|| black_box(mathsolver_core::parse_equation("x^4 + 2*x^3 - 5*x^2 + 3*x - 7 = 0")));
    // });

    group.finish();
}

/// Benchmark solver operations.
fn bench_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("solving");

    // TODO: Add solver benchmarks when solver is implemented
    // group.bench_function("solve_linear", |b| {
    //     let eq = mathsolver_core::parse_equation("2*x + 3 = 7").unwrap();
    //     let solver = mathsolver_core::SmartSolver::new();
    //     let var = mathsolver_core::Variable::new("x");
    //     b.iter(|| black_box(solver.solve(&eq, &var)));
    // });

    // group.bench_function("solve_quadratic", |b| {
    //     let eq = mathsolver_core::parse_equation("x^2 - 5*x + 6 = 0").unwrap();
    //     let solver = mathsolver_core::SmartSolver::new();
    //     let var = mathsolver_core::Variable::new("x");
    //     b.iter(|| black_box(solver.solve(&eq, &var)));
    // });

    group.finish();
}

/// Benchmark numerical methods.
fn bench_numerical(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical");

    // TODO: Add numerical solver benchmarks when implemented
    // group.bench_function("newton_raphson", |b| {
    //     let eq = mathsolver_core::parse_equation("x^2 - 2 = 0").unwrap();
    //     let solver = mathsolver_core::numerical::NewtonRaphson::with_default_config();
    //     let var = mathsolver_core::Variable::new("x");
    //     b.iter(|| black_box(solver.solve(&eq, &var)));
    // });

    group.finish();
}

/// Benchmark expression evaluation.
fn bench_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation");

    // TODO: Add evaluation benchmarks when evaluator is implemented
    // group.bench_function("evaluate_simple", |b| {
    //     let expr = mathsolver_core::parse_expression("2 * x + 3").unwrap();
    //     let mut evaluator = mathsolver_core::numerical::Evaluator::new();
    //     evaluator.set_variable(mathsolver_core::Variable::new("x"), 5.0);
    //     b.iter(|| black_box(evaluator.evaluate(&expr)));
    // });

    // group.bench_function("evaluate_complex", |b| {
    //     let expr = mathsolver_core::parse_expression("sin(x) + cos(x) * tan(x)").unwrap();
    //     let mut evaluator = mathsolver_core::numerical::Evaluator::new();
    //     evaluator.set_variable(mathsolver_core::Variable::new("x"), 1.5);
    //     b.iter(|| black_box(evaluator.evaluate(&expr)));
    // });

    group.finish();
}

/// Benchmark complex number operations.
fn bench_complex_operations(c: &mut Criterion) {
    use mathsolver_core::transforms::ComplexOps;
    use num_complex::Complex64;

    let mut group = c.benchmark_group("complex_operations");

    group.bench_function("complex_to_polar", |b| {
        let c = Complex64::new(3.0, 4.0);
        b.iter(|| black_box(ComplexOps::to_polar(c)));
    });

    group.bench_function("complex_power", |b| {
        let c = Complex64::new(2.0, 3.0);
        b.iter(|| black_box(ComplexOps::de_moivre(c, 5.0)));
    });

    group.finish();
}

/// Benchmark dimension operations.
fn bench_dimensions(c: &mut Criterion) {
    use mathsolver_core::dimensions::{BaseDimension, Dimension};

    let mut group = c.benchmark_group("dimensions");

    group.bench_function("dimension_multiply", |b| {
        let length = Dimension::from_base(BaseDimension::Length, 1);
        let time = Dimension::from_base(BaseDimension::Time, -1);
        b.iter(|| black_box(length.multiply(&time)));
    });

    group.bench_function("dimension_power", |b| {
        let length = Dimension::from_base(BaseDimension::Length, 1);
        b.iter(|| black_box(length.power(2)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_coordinate_transforms,
    bench_parsing,
    bench_solving,
    bench_numerical,
    bench_evaluation,
    bench_complex_operations,
    bench_dimensions,
);

criterion_main!(benches);

// TODO: Add benchmarks for large equation systems
// TODO: Add benchmarks for worst-case scenarios
// TODO: Add memory allocation benchmarks
// TODO: Add comparison benchmarks with other math libraries
// TODO: Add FFI call overhead benchmarks
// TODO: Add parallel solving benchmarks
