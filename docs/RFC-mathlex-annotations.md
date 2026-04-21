# RFC — mathlex annotations substrate (M-R1 … M-R6)

Status: **draft**
Authors: Chris, Claude (drafted autonomously per session 8 decisions)
Target mathlex version: **v0.4.0** (annotations-enabled)
Tracked under task-master tag: `mathlex-upstream`
Cross-reference: `mathcore-units-crate` tag (shared utility crate)

---

## Motivation

thales needs to reason about units, domains, and named constants **without** polluting the math body of `Expression`. The current `Expression` has no side-channel for this metadata. At the ecosystem level, the same metadata is useful to `mathlex-eval` (numeric evaluation), future `thales-quantum`, `thales-qft`, and `thales-GR`. The solution is an **additive, optional, typed, extensible** annotation substrate on `Expression`.

Requirements are numbered **M-R1** … **M-R6**. All are optional at the consumer side; legacy consumers that ignore `annotations` behave identically to today.

## Shared types

All annotation payloads reference types from a new shared crate **`mathcore-units`** (tracked separately). Types consumed:

- `mathcore_units::Unit` — composed unit with scale factor.
- `mathcore_units::Dimension` — 10-vector of base dimensions.
- `mathcore_units::ConstantId` — enum of named constants.
- `thales::api::domain::DomainExpr` — domain algebra (base × qualifier × interval / finite-set / union / intersection / complement).

Mathlex depends on `mathcore-units` but **not** on thales. The `DomainExpr` type must therefore also live in `mathcore-units` (or a similar shared crate) so mathlex can construct it during parsing. **Open question:** whether `DomainExpr` belongs in `mathcore-units` or in a separate `mathcore-algebra` crate. Initial proposal: put it in `mathcore-units` for simplicity; split later if the crate grows too broad.

---

## M-R1 — `AnnotationSet` substrate

Add a field to every `Expression` node (or equivalent — see compatibility notes):

```rust
pub struct Expression {
    // ... existing fields
    pub annotations: AnnotationSet,
}

#[derive(Default, Clone, Debug)]
pub struct AnnotationSet {
    entries: HashMap<AnnotationKind, AnnotationValue>,
}

pub enum AnnotationKind {
    Unit,
    Domain,
    Constant,
    Label,
    // reserved for future: Provenance, SourceSpan, PrecisionHint, ...
}

pub enum AnnotationValue {
    Unit(mathcore_units::Unit),
    Domain(DomainExpr),
    Constant(mathcore_units::ConstantId),
    Label(String),
}
```

### Rationale

- **Typed**: each kind maps to a specific payload type. No stringly-typed metadata.
- **Additive**: new `AnnotationKind` variants can be added without breaking existing consumers. Unknown kinds (if the enum becomes `#[non_exhaustive]`) are ignored by older consumers.
- **Extensible**: consumers that want richer metadata (e.g. a thales-GR extension with metric signature) can define their own annotation kinds in a fork of `AnnotationKind` or via a reserved `Custom(TypeId, Box<dyn Any>)` variant. **Decision pending**: strict enum vs. `Custom` escape hatch.
- **Optional**: `AnnotationSet::default()` is empty. Expressions without annotations behave identically to today.

### Placement

Annotations attach to **any** expression node, not just symbols. Rationale:
- `m[kg] * c^2` — unit on symbol `m`.
- `(x+y)[ℝ]` — domain on a subexpression.
- `E = m c²` with `E` annotated `Joule` — unit on a computed quantity.

### Round-trip invariant

`parse(to_string_with_annotations(expr)) == expr` — full round-trip lossless. See M-R6 for serialization.

---

## M-R2 — Unit annotations (explicit map form)

**Revised per session 8 decision: no inline bracket syntax.**

Units attach via explicit annotations during or after parsing:

```rust
let (expr, meta) = mathlex::parse("E = m * c^2")?;
let with_units = expr
    .annotate_symbol("m", Annotation::Unit(Unit::kilogram()))
    .annotate_symbol("c", Annotation::Constant(ConstantId::SpeedOfLight));
```

No syntax like `m[kg]` or `m{kg}` is added to the mathlex language body. All units are applied as a post-parse operation or via a sidecar annotation block (M-R6).

### Unit dictionary

Mathlex does **not** own the unit dictionary. The finite set of recognized units lives in `mathcore-units::catalog`. Mathlex's role is limited to:

1. Accepting `Unit` values when the caller supplies them.
2. (Optional) Recognizing conventional unit abbreviations in the sidecar annotation block (M-R6) and resolving them against `mathcore_units::catalog::lookup_unit(abbr)`.

### Composed units

`kg·m/s²` (Newton) is expressed as `Unit::composed(&[(kg, 1), (m, 1), (s, -2)])` or equivalently `Unit::from_named("N")` if the catalog resolves by name. Both are equivalent; canonical form is determined by `Unit::normalize()`.

---

## M-R3 — Constant identification

Auto-identify **only** convention-unambiguous symbols:

```rust
// Resolved automatically without annotation:
"hbar", "ħ"           → ConstantId::HBar
"k_B", "kB"            → ConstantId::BoltzmannConstant
"N_A", "Navo"          → ConstantId::AvogadroNumber
"G"                    → ConstantId::GravitationalConstant
"M_earth", "M_⊕"       → ConstantId::EarthMass
"M_sun", "M_☉"         → ConstantId::SolarMass
"epsilon_0", "ε_0"     → ConstantId::VacuumPermittivity
"mu_0", "μ_0"          → ConstantId::VacuumPermeability
"e_charge", "q_e"      → ConstantId::ElementaryCharge
"R_gas"                → ConstantId::GasConstant
"F_faraday"            → ConstantId::FaradayConstant
// ... full list in mathcore_units::catalog::CONVENTIONAL_CONSTANT_SYMBOLS
```

Ambiguous bare letters are **never** auto-resolved:

| Symbol | Possible meanings | Resolution |
|---|---|---|
| `c` | speed of light, constant of integration, generic variable | Requires explicit `Annotation::Constant(ConstantId::SpeedOfLight)` |
| `h` | Planck constant, step size, generic function, height | Requires explicit annotation |
| `e` | Euler's number (already reserved in `Expression::E`), elementary charge | Euler's number wins via reserved symbol; elementary charge via explicit `ConstantId::ElementaryCharge` |
| `i`, `j`, `k` | imaginary unit (reserved), quaternion units (reserved in Quaternion domain), generic index | Reserved for imaginary/quaternion; never auto-resolve to anything else |
| `g` | gravitational acceleration, generic function | Requires explicit `ConstantId::StandardGravity` |
| `α`, `β`, `γ` | fine-structure, gamma factor, generic | Requires explicit annotation |
| `π`, `τ` | Pi, Tau | Already reserved constants in `Expression` |

### Ambiguity policy

When mathlex encounters a conventional symbol in an **ambiguous context** (e.g. `hbar` used as a local variable name inside an integration step), the parser:

1. Applies the auto-annotation by default.
2. Accepts an explicit `Annotation::Label("local")` or `Annotation::Constant(ConstantId::None)` to suppress.

**Decision pending**: default policy. Recommended: auto-annotate, caller overrides if needed.

---

## M-R4 — Domain annotations via standard math notation

Parse standard set-membership notation into domain annotations:

```
x ∈ ℝ            → DomainExpr::Base(Domain { base: Real, qualifier: full })
x ∈ ℝ⁺           → qualifier = { zero: Excluded, positive: true, negative: false, nonzero: true }
x ∈ ℝ⁺₀          → qualifier = { zero: Allowed,  positive: true, negative: false, nonzero: false }
x ∈ ℝ*           → qualifier = { zero: Excluded, positive: true, negative: true,  nonzero: true }
x ∈ [0, 1]       → DomainExpr::Interval { domain: Real, lower: Closed(0), upper: Closed(1) }
x ∈ (0, ∞)       → DomainExpr::Interval { domain: Real, lower: Open(0),   upper: PosInf }
x ∈ ℤ            → DomainExpr::Base(Integer)
x ∈ {1, 2, 3}    → DomainExpr::FiniteSet { domain: Integer, elements: vec![1, 2, 3] }
x ∈ ℝ ∪ ℂ        → DomainExpr::Union(...)
x ∈ ℝ ∩ (0, ∞)   → DomainExpr::Intersection(...)
```

### ASCII aliases

| Unicode | ASCII | Meaning |
|---|---|---|
| `ℕ` | `\N` or `Nat` | Natural numbers |
| `ℤ` | `\Z` or `Int` | Integers |
| `ℚ` | `\Q` or `Rat` | Rationals |
| `ℝ` | `\R` or `Real` | Reals |
| `ℂ` | `\C` or `Complex` | Complex |
| `ℍ` | `\H` or `Quat` | Quaternions |
| `∈` | `in` | Set membership |
| `∪` | `\cup` or `union` | Union |
| `∩` | `\cap` or `intersect` | Intersection |
| `⁺` | `^+` | Positive qualifier |
| `⁻` | `^-` | Negative qualifier |
| `*` | `^*` | Nonzero qualifier |
| `₀` | `_0` | Zero-allowed qualifier |
| `∞` | `\infty` or `inf` | Infinity bound |

### Placement in source

Domain membership statements parse as annotations on the named symbol, not as regular math:

```
// Input
"x ∈ ℝ⁺; y = sqrt(x) + ln(x)"

// Parsed
Expression::Equals(
    Symbol("y") [annotations: {}],
    Add(
        Func(Sqrt, Symbol("x") [annotations: {Domain: ℝ⁺}]),
        Func(Ln,   Symbol("x") [annotations: {Domain: ℝ⁺}])
    )
)
```

The annotation attaches to **every occurrence** of the named symbol in the same parsing scope. Multiple `x ∈ ...` statements for the same symbol compose via intersection (default) per thales `DomainPolicy`.

### `DomainPolicy` at parse time

Mathlex does not enforce domain consistency — thales does that during computation. Mathlex only records declared domains; conflicting declarations become multiple annotations on the same symbol, and thales decides (intersect vs. error) per Request policy.

---

## M-R5 — Backward compatibility

### Wire-level compatibility

Add `annotations: AnnotationSet` to `Expression` with `#[serde(default, skip_serializing_if = "AnnotationSet::is_empty")]`. Existing serialized expressions without this field deserialize to `AnnotationSet::default()` (empty).

### API-level compatibility

Existing mathlex API functions (`parse`, `to_latex`, `to_string`, etc.) continue to work unchanged. Annotation-aware variants are additive:

```rust
// Existing (unchanged)
pub fn parse(input: &str) -> Result<Expression, ParseError>;
pub fn to_string(expr: &Expression) -> String;
pub fn to_latex(expr: &Expression) -> String;

// New (additive)
pub fn parse_with_annotations(input: &str) -> Result<(Expression, AnnotationSet), ParseError>;
pub fn to_string_with_annotations(expr: &Expression, mode: SerializationMode) -> String;
pub fn to_latex_with_annotations(expr: &Expression, companion: &mut Vec<CompanionText>) -> String;
```

The existing `parse` function yields `Expression` with whatever annotations it auto-recognized (conventional constants, inline domain statements); it does not lose information. `parse_with_annotations` is a clarity alias.

### Test requirement

Round-trip tests added to mathlex:

```rust
#[test]
fn legacy_expression_empty_annotations() {
    let expr = parse("x + 1").unwrap();
    assert!(expr.annotations.is_empty());
}

#[test]
fn roundtrip_without_annotations_is_identity() {
    let input = "x + sin(y) * 2";
    let expr = parse(input).unwrap();
    assert_eq!(to_string(&expr), input);
}
```

---

## M-R6 — Serialization modes

Two string serialization modes. Switchable per call:

### Mode (a) — Sidecar (pure math string)

```rust
pub enum SerializationMode {
    Sidecar,
    SeparatorBlock,
}

let s = to_string_with_annotations(&expr, SerializationMode::Sidecar);
// → "E = m * c^2"
```

Annotations are **not** serialized into the string. Caller must carry them externally (e.g. as a `(String, AnnotationSet)` pair, or JSON with two fields).

**Use when:** feeding a legacy consumer, producing LaTeX, producing user-facing display where annotations are rendered as companion text.

### Mode (b) — Separator block

```rust
let s = to_string_with_annotations(&expr, SerializationMode::SeparatorBlock);
// → "E = m * c^2 | m:[kg], c:speed_of_light"
```

Annotations serialize after a `|` separator as a comma-separated list of `symbol:annotation` entries. Round-trippable: `parse` recognizes the `|` separator and restores annotations.

**Syntax:**

```
<math body> | <annotation>, <annotation>, ...
<annotation> := <symbol>:<kind_payload>
<kind_payload> := <unit> | <domain_expr> | <constant_name> | label(<text>)
```

**Use when:** round-trippable transport, storing expressions with their metadata in a single string, debugging.

### LaTeX serialization

LaTeX output is always **pure math** — annotations never appear inside LaTeX. A companion structure is produced separately:

```rust
pub struct CompanionText {
    pub symbol: String,
    pub context: CompanionContext, // Unit | Domain | Constant | Label
    pub rendered_md: String,       // e.g. "where $m$ is measured in kilograms"
}

let mut companions = Vec::new();
let latex = to_latex_with_annotations(&expr, &mut companions);
// latex = "E = m \cdot c^2"
// companions = [
//   { symbol: "m", context: Unit,     rendered_md: "where $m$ is measured in kilograms" },
//   { symbol: "c", context: Constant, rendered_md: "where $c$ is the speed of light" },
// ]
```

The caller (typically thales narrative layer, or a user's display code) decides how to render companions: below the equation, in a footnote, inline parenthetical, etc.

---

## Non-requirements (explicit)

The following are **not** in scope for M-R1…M-R6:

- **Currency units** — rejected per session 8 (non-constant conversion rates). Fintech is out of scope for `mathcore-units`.
- **Inline unit syntax** (`m[kg]`) — rejected per session 8 in favor of explicit annotations.
- **Unit inference** — mathlex does not infer units from context. Inference is a thales computation, not a parsing concern.
- **Constant value substitution** — mathlex never substitutes numeric values for constants. Values live in thales/mathcore-units; mathlex only records the `ConstantId`.
- **Unit arithmetic** — mathlex does not compose or simplify units. Unit normalization lives in `mathcore-units`.

---

## Migration plan

Staged rollout on the `mathlex-upstream` task-master tag:

1. **Phase 1** — `mathcore-units` crate published (stub with minimal types).
2. **Phase 2** — mathlex adds `annotations: AnnotationSet` field + M-R5 compatibility tests. No parser changes yet.
3. **Phase 3** — M-R3 (conventional constant auto-recognition). Pure lookup-table addition; no grammar change.
4. **Phase 4** — M-R4 (domain annotation parsing). Grammar extension for `∈` syntax.
5. **Phase 5** — M-R6 serialization modes.
6. **Phase 6** — M-R2 API (explicit unit annotations helpers). No grammar change.

Phases 2–6 are each a minor version bump of mathlex. thales consumes gradually: v0.8.1 uses ambient domain only, v0.10.0 consumes all annotations.

---

## Open decisions (captured, deferred)

1. `AnnotationKind::Custom` escape hatch vs. strict enum.
2. `DomainExpr` home: `mathcore-units` vs. separate `mathcore-algebra` crate.
3. Default policy for ambiguous conventional-symbol identification (auto-annotate vs. require explicit).
4. Separator-block grammar details: exact characters for the separator, escaping of `|` inside annotations.

---

## References

- thales project CLAUDE.md — architecture rules (Rules 3–6 motivate the annotation substrate).
- thales session-8 handover (2026-04-20) — design decisions.
- CODATA 2022 — source for constant values (public domain, NIST).
- SI Brochure 9th edition (2019) — unit system reference.

## Revision history

- 2026-04-20 draft-0 — initial draft (Claude, session 8, autonomous).
- 2026-04-22 draft-1 — promoted from `.taskmaster/docs/` to public
  `docs/` alongside task T12 completion. Content unchanged from draft-0;
  circulation widened so the mathlex upstream discussion can reference a
  stable URL under the thales repository. Follow-up revisions tracked in
  the `mathlex-upstream` task-master tag.
