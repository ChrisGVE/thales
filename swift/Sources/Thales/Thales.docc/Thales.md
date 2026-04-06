# ``Thales``

A Computer Algebra System (CAS) library for symbolic mathematics in Swift.

## Overview

Thales provides comprehensive symbolic mathematics capabilities for iOS and macOS applications, including:

- **Equation Solving** - Linear, quadratic, polynomial, and systems of equations
- **Calculus** - Differentiation, integration, limits, and series expansions
- **ODE Solvers** - First-order separable and linear ordinary differential equations
- **Series** - Taylor, Maclaurin, Laurent, and asymptotic expansions
- **Coordinate Systems** - 2D/3D transformations and complex numbers
- **Special Functions** - Gamma, beta, erf, and erfc
- **Numerical Methods** - Root finding and optimization when symbolic methods fail

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

### Series Expansions

```swift
// Taylor series of sin(x) around 0 to order 5
let series = try Thales.taylorSeries("sin(x)", variable: "x", center: 0.0, order: 5)
print(series) // x - x^3/6 + x^5/120

// Maclaurin series of e^x
let mac = try Thales.maclaurinSeries("exp(x)", variable: "x", order: 4)
```

### ODE Solving

```swift
// Solve a separable ODE: dy/dx = y
let ode = try Thales.solveODE("dy/dx = y", for: "y", withRespectTo: "x")
print(ode.solution) // "C * exp(x)"

// Solve with initial conditions
let ivp = try Thales.solveODEIVP(
    "dy/dx = y", for: "y", withRespectTo: "x",
    initialConditions: ["y(0)": 1.0]
)
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

### ODE Solvers
- ``Thales/solveODE(_:for:withRespectTo:)``
- ``Thales/solveODEIVP(_:for:withRespectTo:initialConditions:)``
- ``ODEResult``

### Series Expansions
- ``Thales/taylorSeries(_:variable:center:order:)``
- ``Thales/maclaurinSeries(_:variable:order:)``
- ``Thales/laurentSeries(_:variable:center:order:)``
- ``Thales/asymptoticSeries(_:variable:direction:)``
- ``AsymptoticDirection``

### Simplification
- ``Thales/simplify(_:)``
- ``Thales/simplifyTrig(_:)``
- ``Thales/partialFractions(numerator:denominator:variable:)``
- ``SimplificationResult``

### Special Functions
- ``Thales/gamma(_:)``
- ``Thales/beta(_:_:)``
- ``Thales/erf(_:)``
- ``Thales/erfc(_:)``
- ``SpecialFunctionResult``

### Coordinate Types
- ``Point2D``
- ``Point3D``
- ``PolarPoint``
- ``SphericalPoint``

### Complex Numbers
- ``Complex``

### Parsing
- ``Thales/parseEquation(_:)``
- ``Thales/parseExpression(_:)``

### LaTeX Support

Thales can parse LaTeX mathematical notation into its expression tree.

**Supported constructs:**

| Category | LaTeX syntax | Examples |
|----------|-------------|---------|
| Fractions | `\frac{num}{denom}` | `\frac{1}{2}`, `\frac{x+1}{y}` |
| Square root | `\sqrt{x}` | `\sqrt{2}`, `\sqrt{x+1}` |
| nth root | `\sqrt[n]{x}` | `\sqrt[3]{8}`, `\sqrt[n]{x}` |
| Superscripts | `x^{n}` or `x^n` | `x^{2}`, `e^{-x}` |
| Subscripts | `x_{n}` or `x_n` | `x_{1}`, `x_{12}` |
| Greek letters | `\alpha`, `\beta`, `\pi`, etc. | `\alpha`, `\theta`, `\pi` |
| Trig functions | `\sin`, `\cos`, `\tan`, etc. | `\sin{x}`, `\cos(\theta)` |
| Log / exp | `\ln`, `\log`, `\log_{b}`, `\exp` | `\ln{x}`, `\log_{2}(8)` |
| Operators | `\cdot`, `\times`, `\div`, `\pm` | `a \cdot b`, `2 \times 3` |
| Integrals | `\int`, `\int_{a}^{b}` | `\int x \, dx`, `\int_{0}^{1} x^2 \, dx` |
| Limits | `\lim_{x \to a}` | `\lim_{x \to 0} \frac{\sin{x}}{x}` |
| Summation | `\sum_{i=a}^{b}` | `\sum_{i=1}^{10} i` |

- ``Thales/parseLatex(_:)``
- ``Thales/toLatex(_:)``

### Evaluation
- ``Thales/evaluate(_:with:)``
- ``EvaluationResult``

### Result Types
- ``SolutionResult``
- ``DerivativeResult``
- ``IntegralResult``
- ``DefiniteIntegralResult``
- ``LimitResult``

### Error Handling
- ``ThalesError``
