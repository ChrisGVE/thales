// ThalesWrapper.swift
// Swift wrapper for the Thales Computer Algebra System
//
// This file provides a clean, documented Swift API that wraps the
// auto-generated FFI bindings from swift-bridge.

import Foundation

// MARK: - Thales Namespace

/// Thales - A Computer Algebra System for Swift
///
/// Thales provides symbolic mathematics capabilities including equation solving,
/// calculus operations, coordinate transformations, and numerical methods.
///
/// ## Overview
///
/// Thales is organized into several functional areas:
/// - **Equation Solving**: Parse and solve algebraic equations
/// - **Calculus**: Differentiation, integration, and limits
/// - **Coordinates**: 2D/3D coordinate system transformations
/// - **Complex Numbers**: Complex arithmetic and polar form
/// - **Simplification**: Algebraic and trigonometric simplification
///
/// ## Example
///
/// ```swift
/// // Solve a simple equation
/// let solution = try Thales.solve("2*x + 5 = 13", for: "x")
/// print(solution) // "x = 4"
///
/// // Differentiate an expression
/// let derivative = try Thales.differentiate("x^3 + 2*x", withRespectTo: "x")
/// print(derivative.expression) // "3*x^2 + 2"
/// ```
///
/// ## Topics
///
/// ### Equation Solving
/// - ``solve(_:for:)``
/// - ``solve(_:for:knownValues:)``
/// - ``solveNumerically(_:for:initialGuess:)``
/// - ``solveSystem(equations:)``
///
/// ### Calculus
/// - ``differentiate(_:withRespectTo:)``
/// - ``integrate(_:withRespectTo:)``
/// - ``definiteIntegral(_:withRespectTo:from:to:)``
/// - ``limit(_:as:approaches:)``
///
/// ### Coordinate Systems
/// - ``Point2D``
/// - ``Point3D``
/// - ``PolarPoint``
/// - ``SphericalPoint``
public enum Thales {

    /// The current version of Thales
    public static let version = "0.3.0"
}

// MARK: - Error Types

/// Errors that can occur during Thales operations
public enum ThalesError: LocalizedError {
    /// Failed to parse the input expression or equation
    case parseError(String)

    /// No solution exists for the given equation
    case noSolution(String)

    /// The operation failed for the given reason
    case operationFailed(String)

    /// Invalid input was provided
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .parseError(let message):
            return "Parse error: \(message)"
        case .noSolution(let message):
            return "No solution: \(message)"
        case .operationFailed(let message):
            return "Operation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}

// MARK: - Equation Solving

public extension Thales {

    /// Solves an equation for a specified variable.
    ///
    /// This method parses the equation string and attempts to solve it symbolically.
    /// It supports linear, quadratic, polynomial, and some transcendental equations.
    ///
    /// - Parameters:
    ///   - equation: The equation to solve (e.g., "2*x + 5 = 13")
    ///   - variable: The variable to solve for (e.g., "x")
    ///
    /// - Returns: A string representation of the solution
    ///
    /// - Throws: ``ThalesError`` if parsing fails or no solution exists
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Linear equation
    /// let solution = try Thales.solve("2*x + 5 = 13", for: "x")
    /// // solution = "x = 4"
    ///
    /// // Quadratic equation
    /// let roots = try Thales.solve("x^2 - 5*x + 6 = 0", for: "x")
    /// // roots = "x = 2 or x = 3"
    /// ```
    ///
    /// - SeeAlso: ``solveNumerically(_:for:initialGuess:)``
    static func solve(_ equation: String, for variable: String) throws -> String {
        do {
            let result = try solve_equation_ffi(equation, variable)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Solves an equation with known variable values.
    ///
    /// Use this method when you have values for some variables and want to solve
    /// for another variable.
    ///
    /// - Parameters:
    ///   - equation: The equation to solve
    ///   - variable: The variable to solve for
    ///   - knownValues: Dictionary of known variable values
    ///
    /// - Returns: A ``SolutionResult`` containing the solution and resolution steps
    ///
    /// - Throws: ``ThalesError`` if the operation fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.solve(
    ///     "F = m * a",
    ///     for: "a",
    ///     knownValues: ["F": 100.0, "m": 20.0]
    /// )
    /// // result.value = 5.0
    /// ```
    static func solve(_ equation: String, for variable: String, knownValues: [String: Double]) throws -> SolutionResult {
        let jsonData = try JSONSerialization.data(withJSONObject: knownValues)
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"

        do {
            let result = try solve_with_values_ffi(equation, variable, jsonString)
            return SolutionResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Solves an equation numerically using root-finding methods.
    ///
    /// When symbolic solving fails or is not applicable, use this method to find
    /// a numerical approximation of the solution. It uses Newton-Raphson and other
    /// numerical methods internally.
    ///
    /// - Parameters:
    ///   - equation: The equation to solve (must equal zero, e.g., "x^2 - 2")
    ///   - variable: The variable to solve for
    ///   - initialGuess: Starting point for the numerical method
    ///
    /// - Returns: The approximate numerical solution
    ///
    /// - Throws: ``ThalesError`` if convergence fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Find sqrt(2) by solving x^2 - 2 = 0
    /// let root = try Thales.solveNumerically("x^2 - 2", for: "x", initialGuess: 1.0)
    /// // root ≈ 1.4142135623730951
    /// ```
    ///
    /// - Note: The equation should be in the form `f(x) = 0`. If your equation is
    ///   `f(x) = g(x)`, rewrite it as `f(x) - g(x) = 0`.
    static func solveNumerically(_ equation: String, for variable: String, initialGuess: Double) throws -> Double {
        do {
            return try solve_numerically_ffi(equation, variable, initialGuess)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Solves a system of equations.
    ///
    /// - Parameter equations: Array of equation strings
    ///
    /// - Returns: JSON string containing the solutions
    ///
    /// - Throws: ``ThalesError`` if the system cannot be solved
    ///
    /// ## Example
    ///
    /// ```swift
    /// let solutions = try Thales.solveSystem(equations: [
    ///     "x + y = 10",
    ///     "x - y = 4"
    /// ])
    /// // x = 7, y = 3
    /// ```
    static func solveSystem(equations: [String]) throws -> String {
        let jsonData = try JSONSerialization.data(withJSONObject: equations)
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "[]"

        do {
            let result = try solve_system_ffi(jsonString)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Solution Result

/// The result of solving an equation, including the solution and resolution steps.
public struct SolutionResult {
    /// The initial expression before solving
    public let initialExpression: String

    /// The solution expression as a string
    public let result: String

    /// The step-by-step resolution path as JSON
    public let stepsJson: String

    /// Whether the solution was successful
    public let success: Bool

    init(ffiResult: ResolutionPathFFI) {
        self.initialExpression = ffiResult.initial_expr.toString()
        self.result = ffiResult.result_expr.toString()
        self.stepsJson = ffiResult.steps_json.toString()
        self.success = ffiResult.success
    }
}

// MARK: - Calculus

public extension Thales {

    /// Computes the symbolic derivative of an expression.
    ///
    /// Differentiates the given expression with respect to the specified variable
    /// using symbolic differentiation rules (power rule, chain rule, product rule, etc.).
    ///
    /// - Parameters:
    ///   - expression: The expression to differentiate (e.g., "x^3 + sin(x)")
    ///   - variable: The variable to differentiate with respect to
    ///
    /// - Returns: A ``DerivativeResult`` containing the derivative and steps
    ///
    /// - Throws: ``ThalesError`` if the expression cannot be differentiated
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.differentiate("x^3 + 2*x^2 - 5*x", withRespectTo: "x")
    /// print(result.expression) // "3*x^2 + 4*x - 5"
    ///
    /// // Trigonometric function
    /// let trig = try Thales.differentiate("sin(x)*cos(x)", withRespectTo: "x")
    /// // Uses product rule: cos(x)^2 - sin(x)^2
    /// ```
    ///
    /// - SeeAlso: ``nthDerivative(_:withRespectTo:order:)``
    static func differentiate(_ expression: String, withRespectTo variable: String) throws -> DerivativeResult {
        do {
            let result = try differentiate_ffi(expression, variable)
            return DerivativeResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Computes the nth derivative of an expression.
    ///
    /// - Parameters:
    ///   - expression: The expression to differentiate
    ///   - variable: The variable to differentiate with respect to
    ///   - order: The order of differentiation (e.g., 2 for second derivative)
    ///
    /// - Returns: A ``DerivativeResult`` containing the nth derivative
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Second derivative of x^4
    /// let result = try Thales.nthDerivative("x^4", withRespectTo: "x", order: 2)
    /// // result.expression = "12*x^2"
    /// ```
    static func nthDerivative(_ expression: String, withRespectTo variable: String, order: UInt32) throws -> DerivativeResult {
        do {
            let result = try differentiate_n_ffi(expression, variable, order)
            return DerivativeResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Computes the symbolic indefinite integral of an expression.
    ///
    /// Integrates the given expression with respect to the specified variable.
    /// The result includes the constant of integration.
    ///
    /// - Parameters:
    ///   - expression: The expression to integrate
    ///   - variable: The variable of integration
    ///
    /// - Returns: An ``IntegralResult`` containing the antiderivative
    ///
    /// - Throws: ``ThalesError`` if the expression cannot be integrated
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.integrate("3*x^2 + 2*x", withRespectTo: "x")
    /// print(result.expression) // "x^3 + x^2 + C"
    /// ```
    ///
    /// - SeeAlso: ``definiteIntegral(_:withRespectTo:from:to:)``
    static func integrate(_ expression: String, withRespectTo variable: String) throws -> IntegralResult {
        do {
            let result = try integrate_ffi(expression, variable)
            return IntegralResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Computes the definite integral of an expression over an interval.
    ///
    /// Evaluates the integral from the lower bound to the upper bound.
    ///
    /// - Parameters:
    ///   - expression: The expression to integrate
    ///   - variable: The variable of integration
    ///   - lower: The lower bound of integration
    ///   - upper: The upper bound of integration
    ///
    /// - Returns: A ``DefiniteIntegralResult`` containing the numerical value
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Integrate x^2 from 0 to 1
    /// let result = try Thales.definiteIntegral("x^2", withRespectTo: "x", from: 0, to: 1)
    /// print(result.value) // 0.333... (1/3)
    /// ```
    static func definiteIntegral(_ expression: String, withRespectTo variable: String, from lower: Double, to upper: Double) throws -> DefiniteIntegralResult {
        do {
            let result = try definite_integral_ffi(expression, variable, lower, upper)
            return DefiniteIntegralResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Computes the limit of an expression as a variable approaches a value.
    ///
    /// Evaluates limits using direct substitution, L'Hôpital's rule for
    /// indeterminate forms (0/0, ∞/∞), and other techniques.
    ///
    /// - Parameters:
    ///   - expression: The expression to evaluate
    ///   - variable: The variable approaching the limit
    ///   - value: The value being approached
    ///
    /// - Returns: A ``LimitResult`` containing the limit value
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Classic limit: sin(x)/x as x → 0
    /// let result = try Thales.limit("sin(x)/x", as: "x", approaches: 0)
    /// print(result.value) // 1.0
    /// ```
    ///
    /// - SeeAlso: ``limitToInfinity(_:as:)``
    static func limit(_ expression: String, as variable: String, approaches value: Double) throws -> LimitResult {
        do {
            let result = try limit_ffi(expression, variable, value)
            return LimitResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Computes the limit of an expression as a variable approaches infinity.
    ///
    /// - Parameters:
    ///   - expression: The expression to evaluate
    ///   - variable: The variable approaching infinity
    ///
    /// - Returns: A ``LimitResult`` containing the limit value
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.limitToInfinity("1/x", as: "x")
    /// print(result.value) // 0.0
    /// ```
    static func limitToInfinity(_ expression: String, as variable: String) throws -> LimitResult {
        do {
            let result = try limit_infinity_ffi(expression, variable)
            return LimitResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Calculus Result Types

/// The result of a differentiation operation.
public struct DerivativeResult {
    /// The original expression
    public let original: String

    /// The variable of differentiation
    public let variable: String

    /// The derivative expression as a string
    public let derivative: String

    /// The derivative in LaTeX notation
    public let derivativeLatex: String

    init(ffiResult: DifferentiationResultFFI) {
        self.original = ffiResult.original.toString()
        self.variable = ffiResult.variable.toString()
        self.derivative = ffiResult.derivative.toString()
        self.derivativeLatex = ffiResult.derivative_latex.toString()
    }
}

/// The result of an indefinite integration operation.
public struct IntegralResult {
    /// The original expression
    public let original: String

    /// The variable of integration
    public let variable: String

    /// The integral expression (includes "+ C")
    public let integral: String

    /// The integral in LaTeX notation
    public let integralLatex: String

    /// Whether the integration was successful
    public let success: Bool

    /// Error message if integration failed
    public let errorMessage: String?

    init(ffiResult: IntegrationResultFFI) {
        self.original = ffiResult.original.toString()
        self.variable = ffiResult.variable.toString()
        self.integral = ffiResult.integral.toString()
        self.integralLatex = ffiResult.integral_latex.toString()
        self.success = ffiResult.success
        let msg = ffiResult.error_message.toString()
        self.errorMessage = msg.isEmpty ? nil : msg
    }
}

/// The result of a definite integration operation.
public struct DefiniteIntegralResult {
    /// The original expression
    public let original: String

    /// The variable of integration
    public let variable: String

    /// The lower bound of integration
    public let lowerBound: Double

    /// The upper bound of integration
    public let upperBound: Double

    /// The symbolic result
    public let value: String

    /// The result in LaTeX notation
    public let valueLatex: String

    /// The numerical value of the integral
    public let numericValue: Double

    /// Whether the integration was successful
    public let success: Bool

    /// Error message if integration failed
    public let errorMessage: String?

    init(ffiResult: DefiniteIntegralResultFFI) {
        self.original = ffiResult.original.toString()
        self.variable = ffiResult.variable.toString()
        self.lowerBound = ffiResult.lower_bound
        self.upperBound = ffiResult.upper_bound
        self.value = ffiResult.value.toString()
        self.valueLatex = ffiResult.value_latex.toString()
        self.numericValue = ffiResult.numeric_value
        self.success = ffiResult.success
        let msg = ffiResult.error_message.toString()
        self.errorMessage = msg.isEmpty ? nil : msg
    }
}

/// The result of a limit computation.
public struct LimitResult {
    /// The original expression
    public let original: String

    /// The variable approaching the limit
    public let variable: String

    /// The value being approached
    public let approaches: String

    /// The symbolic limit value
    public let value: String

    /// The limit value in LaTeX notation
    public let valueLatex: String

    /// The numerical limit value (may be infinity)
    public let numericValue: Double

    /// Whether the limit computation was successful
    public let success: Bool

    /// Error message if limit computation failed
    public let errorMessage: String?

    init(ffiResult: LimitResultFFI) {
        self.original = ffiResult.original.toString()
        self.variable = ffiResult.variable.toString()
        self.approaches = ffiResult.approaches.toString()
        self.value = ffiResult.value.toString()
        self.valueLatex = ffiResult.value_latex.toString()
        self.numericValue = ffiResult.numeric_value
        self.success = ffiResult.success
        let msg = ffiResult.error_message.toString()
        self.errorMessage = msg.isEmpty ? nil : msg
    }
}

// MARK: - Coordinate Systems

/// A point in 2D Cartesian coordinates.
public struct Point2D: Equatable {
    /// The x-coordinate
    public let x: Double

    /// The y-coordinate
    public let y: Double

    /// Creates a 2D point.
    ///
    /// - Parameters:
    ///   - x: The x-coordinate
    ///   - y: The y-coordinate
    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    /// Converts this Cartesian point to polar coordinates.
    ///
    /// - Returns: The equivalent ``PolarPoint``
    ///
    /// ## Example
    ///
    /// ```swift
    /// let cartesian = Point2D(x: 3, y: 4)
    /// let polar = cartesian.toPolar()
    /// // polar.r = 5.0, polar.theta ≈ 0.927 radians
    /// ```
    public func toPolar() -> PolarPoint {
        let result = cartesian_to_polar_ffi(x, y)
        return PolarPoint(r: result.r, theta: result.theta)
    }
}

/// A point in 3D Cartesian coordinates.
public struct Point3D: Equatable {
    /// The x-coordinate
    public let x: Double

    /// The y-coordinate
    public let y: Double

    /// The z-coordinate
    public let z: Double

    /// Creates a 3D point.
    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    /// Converts this Cartesian point to spherical coordinates.
    ///
    /// - Returns: The equivalent ``SphericalPoint``
    ///
    /// ## Coordinate Convention
    ///
    /// - `r`: Radial distance from origin
    /// - `theta`: Polar angle from positive z-axis (0 to π)
    /// - `phi`: Azimuthal angle in xy-plane from positive x-axis
    public func toSpherical() -> SphericalPoint {
        let result = cartesian_to_spherical_ffi(x, y, z)
        return SphericalPoint(r: result.r, theta: result.theta, phi: result.phi)
    }
}

/// A point in 2D polar coordinates.
public struct PolarPoint: Equatable {
    /// The radial distance from the origin
    public let r: Double

    /// The angle in radians from the positive x-axis
    public let theta: Double

    /// Creates a polar point.
    ///
    /// - Parameters:
    ///   - r: The radial distance (must be non-negative)
    ///   - theta: The angle in radians
    public init(r: Double, theta: Double) {
        self.r = r
        self.theta = theta
    }

    /// Converts this polar point to Cartesian coordinates.
    ///
    /// - Returns: The equivalent ``Point2D``
    ///
    /// ## Example
    ///
    /// ```swift
    /// let polar = PolarPoint(r: 5, theta: .pi / 4)
    /// let cartesian = polar.toCartesian()
    /// // cartesian.x ≈ 3.536, cartesian.y ≈ 3.536
    /// ```
    public func toCartesian() -> Point2D {
        let result = polar_to_cartesian_ffi(r, theta)
        return Point2D(x: result.x, y: result.y)
    }
}

/// A point in 3D spherical coordinates.
public struct SphericalPoint: Equatable {
    /// The radial distance from the origin
    public let r: Double

    /// The polar angle from the positive z-axis (0 to π)
    public let theta: Double

    /// The azimuthal angle in the xy-plane from the positive x-axis
    public let phi: Double

    /// Creates a spherical point.
    public init(r: Double, theta: Double, phi: Double) {
        self.r = r
        self.theta = theta
        self.phi = phi
    }

    /// Converts this spherical point to Cartesian coordinates.
    ///
    /// - Returns: The equivalent ``Point3D``
    public func toCartesian() -> Point3D {
        let result = spherical_to_cartesian_ffi(r, theta, phi)
        return Point3D(x: result.x, y: result.y, z: result.z)
    }
}

// MARK: - Complex Numbers

/// A complex number with real and imaginary parts.
///
/// Complex numbers are of the form `a + bi` where `a` is the real part,
/// `b` is the imaginary part, and `i` is the imaginary unit (√-1).
///
/// ## Example
///
/// ```swift
/// let z1 = Complex(real: 3, imaginary: 4)
/// let z2 = Complex(real: 1, imaginary: 2)
///
/// let sum = z1 + z2          // 4 + 6i
/// let product = z1 * z2      // -5 + 10i
/// let polar = z1.toPolar()   // r = 5, θ ≈ 0.927
/// ```
public struct Complex: Equatable {
    /// The real part of the complex number
    public let real: Double

    /// The imaginary part of the complex number
    public let imaginary: Double

    /// Creates a complex number.
    ///
    /// - Parameters:
    ///   - real: The real part
    ///   - imaginary: The imaginary part
    public init(real: Double, imaginary: Double) {
        self.real = real
        self.imaginary = imaginary
    }

    /// The modulus (absolute value) of the complex number.
    ///
    /// Computed as `√(real² + imaginary²)`.
    public var modulus: Double {
        sqrt(real * real + imaginary * imaginary)
    }

    /// The argument (phase angle) of the complex number in radians.
    ///
    /// Computed as `atan2(imaginary, real)`.
    public var argument: Double {
        atan2(imaginary, real)
    }

    /// The complex conjugate.
    ///
    /// For `a + bi`, returns `a - bi`.
    public var conjugate: Complex {
        Complex(real: real, imaginary: -imaginary)
    }

    /// Converts to polar form.
    ///
    /// - Returns: A ``PolarPoint`` with `r` = modulus and `theta` = argument
    public func toPolar() -> PolarPoint {
        let result = complex_to_polar_ffi(real, imaginary)
        return PolarPoint(r: result.r, theta: result.theta)
    }

    /// Raises the complex number to a power using De Moivre's theorem.
    ///
    /// For a complex number in polar form `r∠θ`, computes `rⁿ∠(nθ)`.
    ///
    /// - Parameter n: The exponent
    /// - Returns: The complex number raised to the power
    ///
    /// ## Example
    ///
    /// ```swift
    /// let z = Complex(real: 1, imaginary: 1)  // √2∠(π/4)
    /// let z2 = z.power(2)                      // 2∠(π/2) = 2i
    /// // z2 ≈ Complex(real: 0, imaginary: 2)
    /// ```
    public func power(_ n: Double) -> Complex {
        let result = complex_power_ffi(real, imaginary, n)
        return Complex(real: result.re, imaginary: result.im)
    }

    /// Adds two complex numbers.
    public static func + (lhs: Complex, rhs: Complex) -> Complex {
        let result = complex_add_ffi(lhs.real, lhs.imaginary, rhs.real, rhs.imaginary)
        return Complex(real: result.re, imaginary: result.im)
    }

    /// Multiplies two complex numbers.
    public static func * (lhs: Complex, rhs: Complex) -> Complex {
        let result = complex_multiply_ffi(lhs.real, lhs.imaginary, rhs.real, rhs.imaginary)
        return Complex(real: result.re, imaginary: result.im)
    }
}

// MARK: - Simplification

public extension Thales {

    /// Simplifies an algebraic expression.
    ///
    /// Applies algebraic simplification rules including:
    /// - Combining like terms
    /// - Reducing fractions
    /// - Applying identities
    ///
    /// - Parameter expression: The expression to simplify
    /// - Returns: A ``SimplificationResult`` with the simplified form
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.simplify("x + x + x")
    /// print(result.expression) // "3*x"
    /// ```
    static func simplify(_ expression: String) throws -> SimplificationResult {
        do {
            let result = try simplify_ffi(expression)
            return SimplificationResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }

    /// Simplifies trigonometric expressions.
    ///
    /// Applies trigonometric identities including:
    /// - Pythagorean identities: sin²x + cos²x = 1
    /// - Double angle formulas
    /// - Product-to-sum rules
    ///
    /// - Parameter expression: The trigonometric expression to simplify
    /// - Returns: A ``SimplificationResult`` with the simplified form
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.simplifyTrig("sin(x)^2 + cos(x)^2")
    /// print(result.expression) // "1"
    /// ```
    static func simplifyTrig(_ expression: String) throws -> SimplificationResult {
        do {
            let result = try simplify_trig_ffi(expression)
            return SimplificationResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

/// The result of a simplification operation.
public struct SimplificationResult {
    /// The original expression
    public let original: String

    /// The simplified expression
    public let simplified: String

    /// The simplified expression in LaTeX notation
    public let simplifiedLatex: String

    init(ffiResult: SimplificationResultFFI) {
        self.original = ffiResult.original.toString()
        self.simplified = ffiResult.simplified.toString()
        self.simplifiedLatex = ffiResult.simplified_latex.toString()
    }
}

// MARK: - LaTeX

public extension Thales {

    /// Parses a LaTeX mathematical expression.
    ///
    /// Supports common LaTeX notation including:
    /// - `\frac{a}{b}` for fractions
    /// - `\sqrt{x}` for square roots
    /// - `\sin`, `\cos`, `\tan`, etc.
    /// - Greek letters: `\alpha`, `\beta`, `\pi`, etc.
    ///
    /// - Parameter latex: The LaTeX string to parse
    /// - Returns: The expression in standard notation
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try Thales.parseLatex("\\frac{x^2 + 1}{x - 1}")
    /// // expr = "(x^2 + 1)/(x - 1)"
    /// ```
    static func parseLatex(_ latex: String) throws -> String {
        do {
            let result = try parse_latex_ffi(latex)
            return result.toString()
        } catch {
            throw ThalesError.parseError(String(describing: error))
        }
    }

    /// Converts an expression to LaTeX notation.
    ///
    /// - Parameter expression: The expression to convert
    /// - Returns: The LaTeX representation
    ///
    /// ## Example
    ///
    /// ```swift
    /// let latex = try Thales.toLatex("(x^2 + 1)/(x - 1)")
    /// // latex = "\\frac{x^{2} + 1}{x - 1}"
    /// ```
    static func toLatex(_ expression: String) throws -> String {
        do {
            let result = try to_latex_ffi(expression)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Evaluation

public extension Thales {

    /// Evaluates an expression with given variable values.
    ///
    /// Substitutes the provided values for variables and computes the result.
    ///
    /// - Parameters:
    ///   - expression: The expression to evaluate
    ///   - values: Dictionary mapping variable names to their values
    ///
    /// - Returns: An ``EvaluationResult`` with the computed value
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.evaluate("x^2 + 2*x + 1", with: ["x": 3.0])
    /// print(result.value) // 16.0
    /// ```
    static func evaluate(_ expression: String, with values: [String: Double]) throws -> EvaluationResult {
        let jsonData = try JSONSerialization.data(withJSONObject: values)
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"

        do {
            let result = try evaluate_ffi(expression, jsonString)
            return EvaluationResult(ffiResult: result)
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

/// The result of an expression evaluation.
public struct EvaluationResult {
    /// The original expression
    public let original: String

    /// The computed numerical value
    public let value: Double

    /// Whether the evaluation was successful
    public let success: Bool

    /// Error message if evaluation failed
    public let errorMessage: String?

    init(ffiResult: EvaluationResultFFI) {
        self.original = ffiResult.original.toString()
        self.value = ffiResult.value
        self.success = ffiResult.success
        let msg = ffiResult.error_message.toString()
        self.errorMessage = msg.isEmpty ? nil : msg
    }
}

// MARK: - Parsing

public extension Thales {

    /// Parses an equation string into the internal representation.
    ///
    /// - Parameter equation: The equation string (e.g., "x + 2 = 5")
    /// - Returns: A string representation of the parsed equation
    /// - Throws: ``ThalesError`` if parsing fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let parsed = try Thales.parseEquation("x^2 - 4 = 0")
    /// // Returns the internal representation
    /// ```
    static func parseEquation(_ equation: String) throws -> String {
        do {
            let result = try parse_equation_ffi(equation)
            return result.toString()
        } catch {
            throw ThalesError.parseError(String(describing: error))
        }
    }

    /// Parses an expression string into the internal representation.
    ///
    /// - Parameter expression: The expression string (e.g., "x^2 + 2*x + 1")
    /// - Returns: A string representation of the parsed expression
    /// - Throws: ``ThalesError`` if parsing fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let parsed = try Thales.parseExpression("sin(x) * cos(x)")
    /// // Returns the internal representation
    /// ```
    static func parseExpression(_ expression: String) throws -> String {
        do {
            let result = try parse_expression_ffi(expression)
            return result.toString()
        } catch {
            throw ThalesError.parseError(String(describing: error))
        }
    }
}

// MARK: - Gradient

public extension Thales {

    /// Computes the gradient of a multivariable function.
    ///
    /// The gradient is a vector of partial derivatives with respect to each variable.
    ///
    /// - Parameters:
    ///   - expression: The expression to differentiate
    ///   - variables: Array of variable names
    ///
    /// - Returns: JSON string containing the gradient components
    ///
    /// - Throws: ``ThalesError`` if the computation fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let gradient = try Thales.gradient("x^2 + y^2 + z^2", variables: ["x", "y", "z"])
    /// // Returns partial derivatives: [2*x, 2*y, 2*z]
    /// ```
    static func gradient(_ expression: String, variables: [String]) throws -> String {
        let jsonData = try JSONSerialization.data(withJSONObject: variables)
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "[]"

        do {
            let result = try gradient_ffi(expression, jsonString)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Inequalities

public extension Thales {

    /// Solves an inequality for a specified variable.
    ///
    /// Supports inequalities with <, >, <=, >=, and chained inequalities.
    ///
    /// - Parameters:
    ///   - inequality: The inequality to solve (e.g., "2*x + 3 < 7")
    ///   - variable: The variable to solve for
    ///
    /// - Returns: A string representation of the solution set
    ///
    /// - Throws: ``ThalesError`` if the inequality cannot be solved
    ///
    /// ## Example
    ///
    /// ```swift
    /// let solution = try Thales.solveInequality("2*x + 3 < 7", for: "x")
    /// // solution = "x < 2"
    ///
    /// let compound = try Thales.solveInequality("1 < 2*x + 1 < 5", for: "x")
    /// // compound = "0 < x < 2"
    /// ```
    static func solveInequality(_ inequality: String, for variable: String) throws -> String {
        do {
            let result = try solve_inequality_ffi(inequality, variable)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Partial Fractions

public extension Thales {

    /// Decomposes a rational expression into partial fractions.
    ///
    /// Useful for integration of rational functions.
    ///
    /// - Parameters:
    ///   - numerator: The numerator polynomial
    ///   - denominator: The denominator polynomial
    ///   - variable: The variable of the rational expression
    ///
    /// - Returns: A string representation of the partial fraction decomposition
    ///
    /// - Throws: ``ThalesError`` if decomposition fails
    ///
    /// ## Example
    ///
    /// ```swift
    /// let result = try Thales.partialFractions(
    ///     numerator: "1",
    ///     denominator: "x^2 - 1",
    ///     variable: "x"
    /// )
    /// // result = "1/(2*(x-1)) - 1/(2*(x+1))"
    /// ```
    static func partialFractions(numerator: String, denominator: String, variable: String) throws -> String {
        do {
            let result = try partial_fractions_ffi(numerator, denominator, variable)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}

// MARK: - Multi-Equation Systems

public extension Thales {

    /// Solves a system of equations with known values and target variables.
    ///
    /// This is a more powerful system solver that accepts known variable values
    /// and can solve for specific target variables.
    ///
    /// - Parameters:
    ///   - equations: Array of equation strings
    ///   - knownValues: Dictionary of known variable values
    ///   - targets: Array of target variable names to solve for
    ///
    /// - Returns: JSON string containing the solutions
    ///
    /// - Throws: ``ThalesError`` if the system cannot be solved
    ///
    /// ## Example
    ///
    /// ```swift
    /// // Physics problem: Find acceleration given force and mass
    /// let solution = try Thales.solveEquationSystem(
    ///     equations: ["F = m * a", "F = 100", "m = 20"],
    ///     knownValues: [:],
    ///     targets: ["a"]
    /// )
    /// // solution contains a = 5
    /// ```
    static func solveEquationSystem(
        equations: [String],
        knownValues: [String: Double],
        targets: [String]
    ) throws -> String {
        let equationsJson = try JSONSerialization.data(withJSONObject: equations)
        let equationsString = String(data: equationsJson, encoding: .utf8) ?? "[]"

        let valuesJson = try JSONSerialization.data(withJSONObject: knownValues)
        let valuesString = String(data: valuesJson, encoding: .utf8) ?? "{}"

        let targetsJson = try JSONSerialization.data(withJSONObject: targets)
        let targetsString = String(data: targetsJson, encoding: .utf8) ?? "[]"

        do {
            let result = try solve_equation_system_ffi(equationsString, valuesString, targetsString)
            return result.toString()
        } catch {
            throw ThalesError.operationFailed(String(describing: error))
        }
    }
}
