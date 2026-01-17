# ``Thales``

A Computer Algebra System (CAS) library for symbolic mathematics in Swift.

## Overview

Thales provides comprehensive symbolic mathematics capabilities for iOS and macOS applications, including:

- **Equation Solving** - Linear, quadratic, polynomial, and systems of equations
- **Calculus** - Differentiation, integration, limits, and series expansions
- **Coordinate Systems** - 2D/3D transformations and complex numbers
- **Numerical Methods** - Root finding when symbolic methods fail

The library is powered by a Rust core via swift-bridge FFI, providing both performance and safety.

## Getting Started

### Solving Equations

```swift
// Solve a linear equation
let result = try Thales.solve("2*x + 5 = 13", for: "x")
print(result) // "x = 4"

// Solve with known values
let result = try Thales.solve(
    "a*x + b = c",
    for: "x",
    knownValues: ["a": 2.0, "b": 5.0, "c": 13.0]
)
```

### Calculus Operations

```swift
// Differentiate
let derivative = try Thales.differentiate("x^3 + 2*x", withRespectTo: "x")
print(derivative.derivative) // "3*x^2 + 2"

// Integrate
let integral = try Thales.integrate("3*x^2", withRespectTo: "x")
print(integral.integral) // "x^3 + C"

// Compute limits
let lim = try Thales.limit("sin(x)/x", as: "x", approaches: 0.0)
print(lim) // "1"
```

### Coordinate Transformations

```swift
// Cartesian to polar
let point = Point2D(x: 3.0, y: 4.0)
let polar = point.toPolar()
print("r = \(polar.r), theta = \(polar.theta)")

// Complex numbers
let z = Complex(real: 1.0, imaginary: 1.0)
let squared = z.power(2)
print("z^2 = \(squared.real) + \(squared.imaginary)i")
```

## Requirements

- iOS 14+ / macOS 11+
- Xcode 14+
- Pre-built `libthales.a` static library (see Installation)

## Topics

### Equation Solving
- ``Thales/solve(_:for:)``
- ``Thales/solve(_:for:knownValues:)``
- ``Thales/solveNumerically(_:for:initialGuess:)``
- ``Thales/solveSystem(equations:)``
- ``Thales/solveEquationSystem(equations:knownValues:targets:)``
- ``Thales/solveInequality(_:for:)``

### Calculus
- ``Thales/differentiate(_:withRespectTo:)``
- ``Thales/nthDerivative(_:withRespectTo:order:)``
- ``Thales/gradient(_:variables:)``
- ``Thales/integrate(_:withRespectTo:)``
- ``Thales/definiteIntegral(_:withRespectTo:from:to:)``
- ``Thales/limit(_:as:approaches:)``
- ``Thales/limitToInfinity(_:as:)``

### Coordinate Types
- ``Point2D``
- ``Point3D``
- ``PolarPoint``
- ``SphericalPoint``

### Complex Numbers
- ``Complex``

### Simplification
- ``Thales/simplify(_:)``
- ``Thales/simplifyTrig(_:)``
- ``Thales/partialFractions(numerator:denominator:variable:)``

### LaTeX Support
- ``Thales/parseLatex(_:)``
- ``Thales/toLatex(_:)``

### Error Handling
- ``ThalesError``
