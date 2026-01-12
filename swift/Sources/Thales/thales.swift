public func parse_equation_ffi<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> RustString {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_equation_ffi(inputAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func parse_expression_ffi<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> RustString {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_expression_ffi(inputAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func solve_equation_ffi<GenericToRustStr: ToRustStr>(_ equation: GenericToRustStr, _ variable: GenericToRustStr) throws -> RustString {
    return variable.toRustStr({ variableAsRustStr in
        return equation.toRustStr({ equationAsRustStr in
        try { let val = __swift_bridge__$solve_equation_ffi(equationAsRustStr, variableAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
    })
}
public func solve_with_values_ffi<GenericToRustStr: ToRustStr>(_ equation: GenericToRustStr, _ variable: GenericToRustStr, _ known_values_json: GenericToRustStr) throws -> ResolutionPathFFI {
    return known_values_json.toRustStr({ known_values_jsonAsRustStr in
        return variable.toRustStr({ variableAsRustStr in
        return equation.toRustStr({ equationAsRustStr in
        try { let val = __swift_bridge__$solve_with_values_ffi(equationAsRustStr, variableAsRustStr, known_values_jsonAsRustStr); switch val.tag { case __swift_bridge__$ResultResolutionPathFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultResolutionPathFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
    })
}
public func solve_numerically_ffi<GenericToRustStr: ToRustStr>(_ equation: GenericToRustStr, _ variable: GenericToRustStr, _ initial_guess: Double) throws -> Double {
    return variable.toRustStr({ variableAsRustStr in
        return equation.toRustStr({ equationAsRustStr in
        try { let val = __swift_bridge__$solve_numerically_ffi(equationAsRustStr, variableAsRustStr, initial_guess); switch val.tag { case __swift_bridge__$ResultF64AndString$ResultOk: return val.payload.ok case __swift_bridge__$ResultF64AndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func cartesian_to_polar_ffi(_ x: Double, _ y: Double) -> PolarCoords {
    __swift_bridge__$cartesian_to_polar_ffi(x, y).intoSwiftRepr()
}
public func polar_to_cartesian_ffi(_ r: Double, _ theta: Double) -> CartesianCoords2D {
    __swift_bridge__$polar_to_cartesian_ffi(r, theta).intoSwiftRepr()
}
public func cartesian_to_spherical_ffi(_ x: Double, _ y: Double, _ z: Double) -> SphericalCoords {
    __swift_bridge__$cartesian_to_spherical_ffi(x, y, z).intoSwiftRepr()
}
public func spherical_to_cartesian_ffi(_ r: Double, _ theta: Double, _ phi: Double) -> CartesianCoords3D {
    __swift_bridge__$spherical_to_cartesian_ffi(r, theta, phi).intoSwiftRepr()
}
public func complex_add_ffi(_ a_re: Double, _ a_im: Double, _ b_re: Double, _ b_im: Double) -> ComplexNumber {
    __swift_bridge__$complex_add_ffi(a_re, a_im, b_re, b_im).intoSwiftRepr()
}
public func complex_multiply_ffi(_ a_re: Double, _ a_im: Double, _ b_re: Double, _ b_im: Double) -> ComplexNumber {
    __swift_bridge__$complex_multiply_ffi(a_re, a_im, b_re, b_im).intoSwiftRepr()
}
public func complex_to_polar_ffi(_ re: Double, _ im: Double) -> PolarCoords {
    __swift_bridge__$complex_to_polar_ffi(re, im).intoSwiftRepr()
}
public func complex_power_ffi(_ re: Double, _ im: Double, _ n: Double) -> ComplexNumber {
    __swift_bridge__$complex_power_ffi(re, im, n).intoSwiftRepr()
}
public func parse_latex_ffi<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> RustString {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_latex_ffi(inputAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func parse_latex_to_latex_ffi<GenericToRustStr: ToRustStr>(_ input: GenericToRustStr) throws -> RustString {
    return input.toRustStr({ inputAsRustStr in
        try { let val = __swift_bridge__$parse_latex_to_latex_ffi(inputAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func to_latex_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr) throws -> RustString {
    return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$to_latex_ffi(expressionAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func differentiate_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr) throws -> DifferentiationResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$differentiate_ffi(expressionAsRustStr, variableAsRustStr); switch val.tag { case __swift_bridge__$ResultDifferentiationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultDifferentiationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func differentiate_n_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr, _ n: UInt32) throws -> DifferentiationResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$differentiate_n_ffi(expressionAsRustStr, variableAsRustStr, n); switch val.tag { case __swift_bridge__$ResultDifferentiationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultDifferentiationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func gradient_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variables_json: GenericToRustStr) throws -> RustString {
    return variables_json.toRustStr({ variables_jsonAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$gradient_ffi(expressionAsRustStr, variables_jsonAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
    })
}
public func integrate_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr) throws -> IntegrationResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$integrate_ffi(expressionAsRustStr, variableAsRustStr); switch val.tag { case __swift_bridge__$ResultIntegrationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultIntegrationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func definite_integral_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr, _ lower: Double, _ upper: Double) throws -> DefiniteIntegralResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$definite_integral_ffi(expressionAsRustStr, variableAsRustStr, lower, upper); switch val.tag { case __swift_bridge__$ResultDefiniteIntegralResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultDefiniteIntegralResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func limit_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr, _ approaches: Double) throws -> LimitResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$limit_ffi(expressionAsRustStr, variableAsRustStr, approaches); switch val.tag { case __swift_bridge__$ResultLimitResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultLimitResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func limit_infinity_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ variable: GenericToRustStr) throws -> LimitResultFFI {
    return variable.toRustStr({ variableAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$limit_infinity_ffi(expressionAsRustStr, variableAsRustStr); switch val.tag { case __swift_bridge__$ResultLimitResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultLimitResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func evaluate_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr, _ values_json: GenericToRustStr) throws -> EvaluationResultFFI {
    return values_json.toRustStr({ values_jsonAsRustStr in
        return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$evaluate_ffi(expressionAsRustStr, values_jsonAsRustStr); switch val.tag { case __swift_bridge__$ResultEvaluationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultEvaluationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
    })
}
public func simplify_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr) throws -> SimplificationResultFFI {
    return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$simplify_ffi(expressionAsRustStr); switch val.tag { case __swift_bridge__$ResultSimplificationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultSimplificationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
}
public func simplify_trig_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr) throws -> SimplificationResultFFI {
    return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$simplify_trig_ffi(expressionAsRustStr); switch val.tag { case __swift_bridge__$ResultSimplificationResultFFIAndString$ResultOk: return val.payload.ok.intoSwiftRepr() case __swift_bridge__$ResultSimplificationResultFFIAndString$ResultErr: throw RustString(ptr: val.payload.err) default: fatalError() } }()
    })
}
public func simplify_trig_with_steps_ffi<GenericToRustStr: ToRustStr>(_ expression: GenericToRustStr) throws -> RustString {
    return expression.toRustStr({ expressionAsRustStr in
        try { let val = __swift_bridge__$simplify_trig_with_steps_ffi(expressionAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func solve_system_ffi<GenericToRustStr: ToRustStr>(_ equations_json: GenericToRustStr) throws -> RustString {
    return equations_json.toRustStr({ equations_jsonAsRustStr in
        try { let val = __swift_bridge__$solve_system_ffi(equations_jsonAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
}
public func solve_inequality_ffi<GenericToRustStr: ToRustStr>(_ inequality: GenericToRustStr, _ variable: GenericToRustStr) throws -> RustString {
    return variable.toRustStr({ variableAsRustStr in
        return inequality.toRustStr({ inequalityAsRustStr in
        try { let val = __swift_bridge__$solve_inequality_ffi(inequalityAsRustStr, variableAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
    })
}
public func partial_fractions_ffi<GenericToRustStr: ToRustStr>(_ numerator: GenericToRustStr, _ denominator: GenericToRustStr, _ variable: GenericToRustStr) throws -> RustString {
    return variable.toRustStr({ variableAsRustStr in
        return denominator.toRustStr({ denominatorAsRustStr in
        return numerator.toRustStr({ numeratorAsRustStr in
        try { let val = __swift_bridge__$partial_fractions_ffi(numeratorAsRustStr, denominatorAsRustStr, variableAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
    })
    })
}
public func solve_equation_system_ffi<GenericToRustStr: ToRustStr>(_ equations_json: GenericToRustStr, _ known_values_json: GenericToRustStr, _ targets_json: GenericToRustStr) throws -> RustString {
    return targets_json.toRustStr({ targets_jsonAsRustStr in
        return known_values_json.toRustStr({ known_values_jsonAsRustStr in
        return equations_json.toRustStr({ equations_jsonAsRustStr in
        try { let val = __swift_bridge__$solve_equation_system_ffi(equations_jsonAsRustStr, known_values_jsonAsRustStr, targets_jsonAsRustStr); if val.is_ok { return RustString(ptr: val.ok_or_err!) } else { throw RustString(ptr: val.ok_or_err!) } }()
    })
    })
    })
}
public struct ResolutionPathFFI {
    public var initial_expr: RustString
    public var steps_json: RustString
    public var result_expr: RustString
    public var success: Bool

    public init(initial_expr: RustString,steps_json: RustString,result_expr: RustString,success: Bool) {
        self.initial_expr = initial_expr
        self.steps_json = steps_json
        self.result_expr = result_expr
        self.success = success
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$ResolutionPathFFI {
        { let val = self; return __swift_bridge__$ResolutionPathFFI(initial_expr: { let rustString = val.initial_expr.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), steps_json: { let rustString = val.steps_json.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), result_expr: { let rustString = val.result_expr.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), success: val.success); }()
    }
}
extension __swift_bridge__$ResolutionPathFFI {
    @inline(__always)
    func intoSwiftRepr() -> ResolutionPathFFI {
        { let val = self; return ResolutionPathFFI(initial_expr: RustString(ptr: val.initial_expr), steps_json: RustString(ptr: val.steps_json), result_expr: RustString(ptr: val.result_expr), success: val.success); }()
    }
}
extension __swift_bridge__$Option$ResolutionPathFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<ResolutionPathFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<ResolutionPathFFI>) -> __swift_bridge__$Option$ResolutionPathFFI {
        if let v = val {
            return __swift_bridge__$Option$ResolutionPathFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$ResolutionPathFFI(is_some: false, val: __swift_bridge__$ResolutionPathFFI())
        }
    }
}
public struct CartesianCoords2D {
    public var x: Double
    public var y: Double

    public init(x: Double,y: Double) {
        self.x = x
        self.y = y
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$CartesianCoords2D {
        { let val = self; return __swift_bridge__$CartesianCoords2D(x: val.x, y: val.y); }()
    }
}
extension __swift_bridge__$CartesianCoords2D {
    @inline(__always)
    func intoSwiftRepr() -> CartesianCoords2D {
        { let val = self; return CartesianCoords2D(x: val.x, y: val.y); }()
    }
}
extension __swift_bridge__$Option$CartesianCoords2D {
    @inline(__always)
    func intoSwiftRepr() -> Optional<CartesianCoords2D> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<CartesianCoords2D>) -> __swift_bridge__$Option$CartesianCoords2D {
        if let v = val {
            return __swift_bridge__$Option$CartesianCoords2D(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$CartesianCoords2D(is_some: false, val: __swift_bridge__$CartesianCoords2D())
        }
    }
}
public struct CartesianCoords3D {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(x: Double,y: Double,z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$CartesianCoords3D {
        { let val = self; return __swift_bridge__$CartesianCoords3D(x: val.x, y: val.y, z: val.z); }()
    }
}
extension __swift_bridge__$CartesianCoords3D {
    @inline(__always)
    func intoSwiftRepr() -> CartesianCoords3D {
        { let val = self; return CartesianCoords3D(x: val.x, y: val.y, z: val.z); }()
    }
}
extension __swift_bridge__$Option$CartesianCoords3D {
    @inline(__always)
    func intoSwiftRepr() -> Optional<CartesianCoords3D> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<CartesianCoords3D>) -> __swift_bridge__$Option$CartesianCoords3D {
        if let v = val {
            return __swift_bridge__$Option$CartesianCoords3D(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$CartesianCoords3D(is_some: false, val: __swift_bridge__$CartesianCoords3D())
        }
    }
}
public struct PolarCoords {
    public var r: Double
    public var theta: Double

    public init(r: Double,theta: Double) {
        self.r = r
        self.theta = theta
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$PolarCoords {
        { let val = self; return __swift_bridge__$PolarCoords(r: val.r, theta: val.theta); }()
    }
}
extension __swift_bridge__$PolarCoords {
    @inline(__always)
    func intoSwiftRepr() -> PolarCoords {
        { let val = self; return PolarCoords(r: val.r, theta: val.theta); }()
    }
}
extension __swift_bridge__$Option$PolarCoords {
    @inline(__always)
    func intoSwiftRepr() -> Optional<PolarCoords> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<PolarCoords>) -> __swift_bridge__$Option$PolarCoords {
        if let v = val {
            return __swift_bridge__$Option$PolarCoords(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$PolarCoords(is_some: false, val: __swift_bridge__$PolarCoords())
        }
    }
}
public struct SphericalCoords {
    public var r: Double
    public var theta: Double
    public var phi: Double

    public init(r: Double,theta: Double,phi: Double) {
        self.r = r
        self.theta = theta
        self.phi = phi
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$SphericalCoords {
        { let val = self; return __swift_bridge__$SphericalCoords(r: val.r, theta: val.theta, phi: val.phi); }()
    }
}
extension __swift_bridge__$SphericalCoords {
    @inline(__always)
    func intoSwiftRepr() -> SphericalCoords {
        { let val = self; return SphericalCoords(r: val.r, theta: val.theta, phi: val.phi); }()
    }
}
extension __swift_bridge__$Option$SphericalCoords {
    @inline(__always)
    func intoSwiftRepr() -> Optional<SphericalCoords> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<SphericalCoords>) -> __swift_bridge__$Option$SphericalCoords {
        if let v = val {
            return __swift_bridge__$Option$SphericalCoords(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$SphericalCoords(is_some: false, val: __swift_bridge__$SphericalCoords())
        }
    }
}
public struct ComplexNumber {
    public var real: Double
    public var imaginary: Double

    public init(real: Double,imaginary: Double) {
        self.real = real
        self.imaginary = imaginary
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$ComplexNumber {
        { let val = self; return __swift_bridge__$ComplexNumber(real: val.real, imaginary: val.imaginary); }()
    }
}
extension __swift_bridge__$ComplexNumber {
    @inline(__always)
    func intoSwiftRepr() -> ComplexNumber {
        { let val = self; return ComplexNumber(real: val.real, imaginary: val.imaginary); }()
    }
}
extension __swift_bridge__$Option$ComplexNumber {
    @inline(__always)
    func intoSwiftRepr() -> Optional<ComplexNumber> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<ComplexNumber>) -> __swift_bridge__$Option$ComplexNumber {
        if let v = val {
            return __swift_bridge__$Option$ComplexNumber(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$ComplexNumber(is_some: false, val: __swift_bridge__$ComplexNumber())
        }
    }
}
public struct DifferentiationResultFFI {
    public var original: RustString
    public var variable: RustString
    public var derivative: RustString
    public var derivative_latex: RustString

    public init(original: RustString,variable: RustString,derivative: RustString,derivative_latex: RustString) {
        self.original = original
        self.variable = variable
        self.derivative = derivative
        self.derivative_latex = derivative_latex
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$DifferentiationResultFFI {
        { let val = self; return __swift_bridge__$DifferentiationResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), variable: { let rustString = val.variable.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), derivative: { let rustString = val.derivative.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), derivative_latex: { let rustString = val.derivative_latex.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$DifferentiationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> DifferentiationResultFFI {
        { let val = self; return DifferentiationResultFFI(original: RustString(ptr: val.original), variable: RustString(ptr: val.variable), derivative: RustString(ptr: val.derivative), derivative_latex: RustString(ptr: val.derivative_latex)); }()
    }
}
extension __swift_bridge__$Option$DifferentiationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<DifferentiationResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<DifferentiationResultFFI>) -> __swift_bridge__$Option$DifferentiationResultFFI {
        if let v = val {
            return __swift_bridge__$Option$DifferentiationResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$DifferentiationResultFFI(is_some: false, val: __swift_bridge__$DifferentiationResultFFI())
        }
    }
}
public struct IntegrationResultFFI {
    public var original: RustString
    public var variable: RustString
    public var integral: RustString
    public var integral_latex: RustString
    public var success: Bool
    public var error_message: RustString

    public init(original: RustString,variable: RustString,integral: RustString,integral_latex: RustString,success: Bool,error_message: RustString) {
        self.original = original
        self.variable = variable
        self.integral = integral
        self.integral_latex = integral_latex
        self.success = success
        self.error_message = error_message
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$IntegrationResultFFI {
        { let val = self; return __swift_bridge__$IntegrationResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), variable: { let rustString = val.variable.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), integral: { let rustString = val.integral.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), integral_latex: { let rustString = val.integral_latex.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), success: val.success, error_message: { let rustString = val.error_message.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$IntegrationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> IntegrationResultFFI {
        { let val = self; return IntegrationResultFFI(original: RustString(ptr: val.original), variable: RustString(ptr: val.variable), integral: RustString(ptr: val.integral), integral_latex: RustString(ptr: val.integral_latex), success: val.success, error_message: RustString(ptr: val.error_message)); }()
    }
}
extension __swift_bridge__$Option$IntegrationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<IntegrationResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<IntegrationResultFFI>) -> __swift_bridge__$Option$IntegrationResultFFI {
        if let v = val {
            return __swift_bridge__$Option$IntegrationResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$IntegrationResultFFI(is_some: false, val: __swift_bridge__$IntegrationResultFFI())
        }
    }
}
public struct DefiniteIntegralResultFFI {
    public var original: RustString
    public var variable: RustString
    public var lower_bound: Double
    public var upper_bound: Double
    public var value: RustString
    public var value_latex: RustString
    public var numeric_value: Double
    public var success: Bool
    public var error_message: RustString

    public init(original: RustString,variable: RustString,lower_bound: Double,upper_bound: Double,value: RustString,value_latex: RustString,numeric_value: Double,success: Bool,error_message: RustString) {
        self.original = original
        self.variable = variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value = value
        self.value_latex = value_latex
        self.numeric_value = numeric_value
        self.success = success
        self.error_message = error_message
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$DefiniteIntegralResultFFI {
        { let val = self; return __swift_bridge__$DefiniteIntegralResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), variable: { let rustString = val.variable.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), lower_bound: val.lower_bound, upper_bound: val.upper_bound, value: { let rustString = val.value.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), value_latex: { let rustString = val.value_latex.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), numeric_value: val.numeric_value, success: val.success, error_message: { let rustString = val.error_message.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$DefiniteIntegralResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> DefiniteIntegralResultFFI {
        { let val = self; return DefiniteIntegralResultFFI(original: RustString(ptr: val.original), variable: RustString(ptr: val.variable), lower_bound: val.lower_bound, upper_bound: val.upper_bound, value: RustString(ptr: val.value), value_latex: RustString(ptr: val.value_latex), numeric_value: val.numeric_value, success: val.success, error_message: RustString(ptr: val.error_message)); }()
    }
}
extension __swift_bridge__$Option$DefiniteIntegralResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<DefiniteIntegralResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<DefiniteIntegralResultFFI>) -> __swift_bridge__$Option$DefiniteIntegralResultFFI {
        if let v = val {
            return __swift_bridge__$Option$DefiniteIntegralResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$DefiniteIntegralResultFFI(is_some: false, val: __swift_bridge__$DefiniteIntegralResultFFI())
        }
    }
}
public struct LimitResultFFI {
    public var original: RustString
    public var variable: RustString
    public var approaches: RustString
    public var value: RustString
    public var value_latex: RustString
    public var numeric_value: Double
    public var success: Bool
    public var error_message: RustString

    public init(original: RustString,variable: RustString,approaches: RustString,value: RustString,value_latex: RustString,numeric_value: Double,success: Bool,error_message: RustString) {
        self.original = original
        self.variable = variable
        self.approaches = approaches
        self.value = value
        self.value_latex = value_latex
        self.numeric_value = numeric_value
        self.success = success
        self.error_message = error_message
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$LimitResultFFI {
        { let val = self; return __swift_bridge__$LimitResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), variable: { let rustString = val.variable.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), approaches: { let rustString = val.approaches.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), value: { let rustString = val.value.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), value_latex: { let rustString = val.value_latex.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), numeric_value: val.numeric_value, success: val.success, error_message: { let rustString = val.error_message.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$LimitResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> LimitResultFFI {
        { let val = self; return LimitResultFFI(original: RustString(ptr: val.original), variable: RustString(ptr: val.variable), approaches: RustString(ptr: val.approaches), value: RustString(ptr: val.value), value_latex: RustString(ptr: val.value_latex), numeric_value: val.numeric_value, success: val.success, error_message: RustString(ptr: val.error_message)); }()
    }
}
extension __swift_bridge__$Option$LimitResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<LimitResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<LimitResultFFI>) -> __swift_bridge__$Option$LimitResultFFI {
        if let v = val {
            return __swift_bridge__$Option$LimitResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$LimitResultFFI(is_some: false, val: __swift_bridge__$LimitResultFFI())
        }
    }
}
public struct EvaluationResultFFI {
    public var original: RustString
    public var value: Double
    public var success: Bool
    public var error_message: RustString

    public init(original: RustString,value: Double,success: Bool,error_message: RustString) {
        self.original = original
        self.value = value
        self.success = success
        self.error_message = error_message
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$EvaluationResultFFI {
        { let val = self; return __swift_bridge__$EvaluationResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), value: val.value, success: val.success, error_message: { let rustString = val.error_message.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$EvaluationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> EvaluationResultFFI {
        { let val = self; return EvaluationResultFFI(original: RustString(ptr: val.original), value: val.value, success: val.success, error_message: RustString(ptr: val.error_message)); }()
    }
}
extension __swift_bridge__$Option$EvaluationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<EvaluationResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<EvaluationResultFFI>) -> __swift_bridge__$Option$EvaluationResultFFI {
        if let v = val {
            return __swift_bridge__$Option$EvaluationResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$EvaluationResultFFI(is_some: false, val: __swift_bridge__$EvaluationResultFFI())
        }
    }
}
public struct SimplificationResultFFI {
    public var original: RustString
    public var simplified: RustString
    public var simplified_latex: RustString

    public init(original: RustString,simplified: RustString,simplified_latex: RustString) {
        self.original = original
        self.simplified = simplified
        self.simplified_latex = simplified_latex
    }

    @inline(__always)
    func intoFfiRepr() -> __swift_bridge__$SimplificationResultFFI {
        { let val = self; return __swift_bridge__$SimplificationResultFFI(original: { let rustString = val.original.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), simplified: { let rustString = val.simplified.intoRustString(); rustString.isOwned = false; return rustString.ptr }(), simplified_latex: { let rustString = val.simplified_latex.intoRustString(); rustString.isOwned = false; return rustString.ptr }()); }()
    }
}
extension __swift_bridge__$SimplificationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> SimplificationResultFFI {
        { let val = self; return SimplificationResultFFI(original: RustString(ptr: val.original), simplified: RustString(ptr: val.simplified), simplified_latex: RustString(ptr: val.simplified_latex)); }()
    }
}
extension __swift_bridge__$Option$SimplificationResultFFI {
    @inline(__always)
    func intoSwiftRepr() -> Optional<SimplificationResultFFI> {
        if self.is_some {
            return self.val.intoSwiftRepr()
        } else {
            return nil
        }
    }

    @inline(__always)
    static func fromSwiftRepr(_ val: Optional<SimplificationResultFFI>) -> __swift_bridge__$Option$SimplificationResultFFI {
        if let v = val {
            return __swift_bridge__$Option$SimplificationResultFFI(is_some: true, val: v.intoFfiRepr())
        } else {
            return __swift_bridge__$Option$SimplificationResultFFI(is_some: false, val: __swift_bridge__$SimplificationResultFFI())
        }
    }
}


