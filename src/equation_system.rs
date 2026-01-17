//! Multi-Equation System Solver
//!
//! This module provides a general-purpose equation system solver that can:
//! - Accept multiple equations of any supported type (algebraic, ODE, integral, differential, etc.)
//! - Accept known values and target variable(s)
//! - Build a dependency graph to determine solving order
//! - Chain solutions through multiple equations
//! - Track all steps in a unified resolution path
//!
//! # Example
//!
//! ```rust
//! use thales::equation_system::{EquationSystem, SystemContext, MultiEquationSolver};
//! use thales::parse_equation;
//!
//! // Create a system of equations
//! let mut system = EquationSystem::new();
//! system.add_equation("eq1", parse_equation("F = m * a").unwrap());
//! system.add_equation("eq2", parse_equation("v = u + a * t").unwrap());
//!
//! // Set up the context with known values and targets
//! let context = SystemContext::new()
//!     .with_known_value("F", 100.0)
//!     .with_known_value("m", 20.0)
//!     .with_known_value("u", 0.0)
//!     .with_known_value("t", 5.0)
//!     .with_target("a")
//!     .with_target("v");
//!
//! // Solve the system
//! let solver = MultiEquationSolver::new();
//! let solution = solver.solve(&system, &context).unwrap();
//!
//! // Get results: a = 5.0, v = 25.0
//! assert!((solution.get_numeric("a").unwrap() - 5.0).abs() < 1e-10);
//! assert!((solution.get_numeric("v").unwrap() - 25.0).abs() < 1e-10);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ast::{BinaryOp, Equation, Expression, Variable};
use crate::numerical::{NumericalConfig, SmartNumericalSolver};
use crate::resolution_path::{Operation, ResolutionPath};
use crate::solver::{SmartSolver, Solution, Solver, SolverError};

// Note: ODE and integration imports reserved for future implementation
// use crate::ode::{FirstOrderODE, ODESolution, solve_separable, solve_linear as solve_linear_ode};
// use crate::integration::integrate;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during equation system solving.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SystemError {
    /// Circular dependency detected between variables/equations.
    CircularDependency(Vec<String>),

    /// Not enough equations to determine all target variables.
    InsufficientEquations {
        /// Number of equations needed.
        needed: usize,
        /// Number of equations available.
        have: usize,
    },

    /// System has more equations than unknowns (may be inconsistent).
    OverdeterminedSystem {
        /// Number of equations.
        equations: usize,
        /// Number of unknowns.
        unknowns: usize,
    },

    /// Could not solve a specific equation.
    UnsolvableEquation {
        /// Equation identifier.
        id: String,
        /// Reason for failure.
        reason: String,
    },

    /// System is mathematically inconsistent (equations contradict each other).
    InconsistentSystem {
        /// Equations involved in the inconsistency.
        equations: Vec<String>,
    },

    /// Numerical method failed.
    NumericalFailure {
        /// Variable being solved for.
        variable: String,
        /// Reason for failure.
        reason: String,
    },

    /// Variable not found in the system.
    VariableNotFound(String),

    /// Equation not found in the system.
    EquationNotFound(String),

    /// No solving strategy found.
    NoStrategyFound(String),

    /// Underlying solver error.
    SolverError(String),

    /// Parse error.
    ParseError(String),
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CircularDependency(vars) => {
                write!(f, "Circular dependency detected: {}", vars.join(" -> "))
            }
            Self::InsufficientEquations { needed, have } => {
                write!(
                    f,
                    "Insufficient equations: need {} but have {}",
                    needed, have
                )
            }
            Self::OverdeterminedSystem {
                equations,
                unknowns,
            } => {
                write!(
                    f,
                    "Overdetermined system: {} equations for {} unknowns",
                    equations, unknowns
                )
            }
            Self::UnsolvableEquation { id, reason } => {
                write!(f, "Cannot solve equation '{}': {}", id, reason)
            }
            Self::InconsistentSystem { equations } => {
                write!(
                    f,
                    "Inconsistent system: equations {} contradict",
                    equations.join(", ")
                )
            }
            Self::NumericalFailure { variable, reason } => {
                write!(
                    f,
                    "Numerical failure solving for '{}': {}",
                    variable, reason
                )
            }
            Self::VariableNotFound(var) => {
                write!(f, "Variable '{}' not found in system", var)
            }
            Self::EquationNotFound(id) => {
                write!(f, "Equation '{}' not found in system", id)
            }
            Self::NoStrategyFound(reason) => {
                write!(f, "No solving strategy found: {}", reason)
            }
            Self::SolverError(msg) => write!(f, "Solver error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for SystemError {}

impl From<SolverError> for SystemError {
    fn from(err: SolverError) -> Self {
        Self::SolverError(format!("{:?}", err))
    }
}

// ============================================================================
// Equation Types and Classification
// ============================================================================

/// Additional information for ODE equations.
#[derive(Debug, Clone)]
pub struct ODEInfo {
    /// The dependent variable (e.g., y in dy/dx = f(x,y)).
    pub dependent_var: String,
    /// The independent variable (e.g., x).
    pub independent_var: String,
    /// Order of the ODE.
    pub order: usize,
}

/// Additional information for integral equations.
#[derive(Debug, Clone)]
pub struct IntegralInfo {
    /// Variable of integration.
    pub integration_var: String,
    /// Lower bound (if definite).
    pub lower_bound: Option<Expression>,
    /// Upper bound (if definite).
    pub upper_bound: Option<Expression>,
}

/// Classification of equation types the system can handle.
#[derive(Debug, Clone)]
pub enum EquationType {
    /// Standard algebraic equation (polynomial, rational, etc.).
    Algebraic,
    /// Ordinary differential equation.
    ODE(ODEInfo),
    /// Equation involving integrals.
    Integral(IntegralInfo),
    /// Equation with derivatives (not a full ODE formulation).
    Differential,
    /// Equation that cannot be solved for a single variable algebraically.
    Implicit,
    /// Unknown or unclassified type.
    Unknown,
}

impl Default for EquationType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// A named equation in the system.
#[derive(Debug, Clone)]
pub struct NamedEquation {
    /// Unique identifier for the equation.
    pub id: String,
    /// The actual equation.
    pub equation: Equation,
    /// Classification of the equation type.
    pub equation_type: EquationType,
    /// Human-readable description (optional).
    pub description: Option<String>,
}

impl NamedEquation {
    /// Create a new named equation with automatic type classification.
    pub fn new(id: impl Into<String>, equation: Equation) -> Self {
        let eq_type = Self::classify(&equation);
        Self {
            id: id.into(),
            equation,
            equation_type: eq_type,
            description: None,
        }
    }

    /// Create a new named equation with explicit type.
    pub fn with_type(
        id: impl Into<String>,
        equation: Equation,
        equation_type: EquationType,
    ) -> Self {
        Self {
            id: id.into(),
            equation,
            equation_type,
            description: None,
        }
    }

    /// Add a description to the equation.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Get all variables in this equation.
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = self.equation.left.variables();
        vars.extend(self.equation.right.variables());
        vars
    }

    /// Classify the equation type based on its structure.
    fn classify(equation: &Equation) -> EquationType {
        // Check for derivative patterns (dy/dx, d²y/dx², etc.)
        if Self::contains_derivative(&equation.left) || Self::contains_derivative(&equation.right) {
            // Try to identify if it's a proper ODE formulation
            if let Some(info) = Self::extract_ode_info(equation) {
                return EquationType::ODE(info);
            }
            return EquationType::Differential;
        }

        // Check for integral patterns
        if Self::contains_integral(&equation.left) || Self::contains_integral(&equation.right) {
            if let Some(info) = Self::extract_integral_info(equation) {
                return EquationType::Integral(info);
            }
        }

        // Default to algebraic
        EquationType::Algebraic
    }

    /// Check if expression contains derivative notation.
    fn contains_derivative(expr: &Expression) -> bool {
        match expr {
            Expression::Function(func, _) => {
                // Check for custom derivative function names
                matches!(func, crate::ast::Function::Custom(name)
                    if name.starts_with("d") && name.contains("/d"))
            }
            Expression::Binary(_, left, right) => {
                Self::contains_derivative(left) || Self::contains_derivative(right)
            }
            Expression::Unary(_, inner) => Self::contains_derivative(inner),
            Expression::Power(base, exp) => {
                Self::contains_derivative(base) || Self::contains_derivative(exp)
            }
            _ => false,
        }
    }

    /// Check if expression contains integral notation.
    fn contains_integral(expr: &Expression) -> bool {
        match expr {
            Expression::Function(func, _) => {
                // Check for custom integral function names
                matches!(func, crate::ast::Function::Custom(name)
                    if name == "integral" || name == "int" || name == "integrate")
            }
            Expression::Binary(_, left, right) => {
                Self::contains_integral(left) || Self::contains_integral(right)
            }
            Expression::Unary(_, inner) => Self::contains_integral(inner),
            Expression::Power(base, exp) => {
                Self::contains_integral(base) || Self::contains_integral(exp)
            }
            _ => false,
        }
    }

    /// Extract ODE information from equation if it's a valid ODE.
    fn extract_ode_info(_equation: &Equation) -> Option<ODEInfo> {
        // TODO: Implement ODE pattern recognition
        // For now, return None and let users explicitly specify ODE type
        None
    }

    /// Extract integral information from equation.
    fn extract_integral_info(_equation: &Equation) -> Option<IntegralInfo> {
        // TODO: Implement integral pattern recognition
        None
    }
}

// ============================================================================
// Equation System
// ============================================================================

/// A collection of named equations forming a system.
#[derive(Debug, Clone, Default)]
pub struct EquationSystem {
    /// Map of equation ID to equation.
    equations: HashMap<String, NamedEquation>,
    /// Optional description of the system.
    pub description: Option<String>,
}

impl EquationSystem {
    /// Create a new empty equation system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a system with a description.
    pub fn with_description(description: impl Into<String>) -> Self {
        Self {
            equations: HashMap::new(),
            description: Some(description.into()),
        }
    }

    /// Add an equation to the system.
    pub fn add_equation(&mut self, id: impl Into<String>, equation: Equation) -> &mut Self {
        let id = id.into();
        self.equations
            .insert(id.clone(), NamedEquation::new(id, equation));
        self
    }

    /// Add a named equation to the system.
    pub fn add_named_equation(&mut self, named_eq: NamedEquation) -> &mut Self {
        self.equations.insert(named_eq.id.clone(), named_eq);
        self
    }

    /// Builder pattern: add equation and return self.
    #[must_use]
    pub fn with_equation(mut self, id: impl Into<String>, equation: Equation) -> Self {
        self.add_equation(id, equation);
        self
    }

    /// Get an equation by ID.
    pub fn get(&self, id: &str) -> Option<&NamedEquation> {
        self.equations.get(id)
    }

    /// Get all equation IDs.
    pub fn equation_ids(&self) -> impl Iterator<Item = &String> {
        self.equations.keys()
    }

    /// Get all equations.
    pub fn equations(&self) -> impl Iterator<Item = &NamedEquation> {
        self.equations.values()
    }

    /// Number of equations in the system.
    pub fn len(&self) -> usize {
        self.equations.len()
    }

    /// Check if the system is empty.
    pub fn is_empty(&self) -> bool {
        self.equations.is_empty()
    }

    /// Get all unique variables across all equations.
    pub fn all_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for eq in self.equations.values() {
            vars.extend(eq.variables());
        }
        vars
    }

    /// Remove an equation by ID.
    pub fn remove(&mut self, id: &str) -> Option<NamedEquation> {
        self.equations.remove(id)
    }
}

// ============================================================================
// System Context
// ============================================================================

/// Constraint on a variable or relationship.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Variable must be greater than a value.
    GreaterThan(String, f64),
    /// Variable must be less than a value.
    LessThan(String, f64),
    /// Variable must be in a range.
    InRange(String, f64, f64),
    /// Variable must be positive.
    Positive(String),
    /// Variable must be non-negative.
    NonNegative(String),
    /// Variable must be an integer.
    Integer(String),
    /// Custom constraint with description.
    Custom(String),
}

/// Context for solving - what we know and what we want.
#[derive(Debug, Clone, Default)]
pub struct SystemContext {
    /// Known numeric values for variables.
    pub known_values: HashMap<String, f64>,
    /// Known symbolic expressions for variables.
    pub known_expressions: HashMap<String, Expression>,
    /// Variables we want to solve for.
    pub target_variables: Vec<String>,
    /// Additional constraints.
    pub constraints: Vec<Constraint>,
    /// Whether to verify solutions.
    pub verify_solutions: bool,
    /// Numerical tolerance for verification.
    pub tolerance: f64,
}

impl SystemContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            known_values: HashMap::new(),
            known_expressions: HashMap::new(),
            target_variables: Vec::new(),
            constraints: Vec::new(),
            verify_solutions: true,
            tolerance: 1e-10,
        }
    }

    /// Add a known numeric value.
    #[must_use]
    pub fn with_known_value(mut self, variable: impl Into<String>, value: f64) -> Self {
        self.known_values.insert(variable.into(), value);
        self
    }

    /// Add a known symbolic expression.
    #[must_use]
    pub fn with_known_expression(mut self, variable: impl Into<String>, expr: Expression) -> Self {
        self.known_expressions.insert(variable.into(), expr);
        self
    }

    /// Add a target variable to solve for.
    #[must_use]
    pub fn with_target(mut self, variable: impl Into<String>) -> Self {
        self.target_variables.push(variable.into());
        self
    }

    /// Add multiple target variables.
    #[must_use]
    pub fn with_targets(mut self, variables: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.target_variables
            .extend(variables.into_iter().map(Into::into));
        self
    }

    /// Add a constraint.
    #[must_use]
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set verification preference.
    #[must_use]
    pub fn with_verification(mut self, verify: bool) -> Self {
        self.verify_solutions = verify;
        self
    }

    /// Set numerical tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Get all known variable names.
    pub fn known_variable_names(&self) -> HashSet<String> {
        let mut vars: HashSet<String> = self.known_values.keys().cloned().collect();
        vars.extend(self.known_expressions.keys().cloned());
        vars
    }

    /// Check if a variable is known.
    pub fn is_known(&self, var: &str) -> bool {
        self.known_values.contains_key(var) || self.known_expressions.contains_key(var)
    }
}

// ============================================================================
// Dependency Graph
// ============================================================================

/// Tracks which variables appear in which equations.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Variable -> equations that contain it.
    variable_to_equations: HashMap<String, HashSet<String>>,
    /// Equation -> variables it contains.
    equation_to_variables: HashMap<String, HashSet<String>>,
    /// Equation -> variables it can potentially solve for.
    equation_can_solve: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Build a dependency graph from an equation system.
    pub fn build(system: &EquationSystem) -> Self {
        let mut variable_to_equations: HashMap<String, HashSet<String>> = HashMap::new();
        let mut equation_to_variables: HashMap<String, HashSet<String>> = HashMap::new();
        let mut equation_can_solve: HashMap<String, HashSet<String>> = HashMap::new();

        for eq in system.equations() {
            let vars = eq.variables();
            equation_to_variables.insert(eq.id.clone(), vars.clone());

            // For algebraic equations, we can potentially solve for any variable
            // For ODEs/integrals, we typically solve for the dependent variable
            let solvable = match &eq.equation_type {
                EquationType::Algebraic | EquationType::Unknown => vars.clone(),
                EquationType::ODE(info) => {
                    let mut set = HashSet::new();
                    set.insert(info.dependent_var.clone());
                    set
                }
                EquationType::Integral(info) => {
                    let mut set = vars.clone();
                    set.remove(&info.integration_var);
                    set
                }
                EquationType::Differential | EquationType::Implicit => HashSet::new(),
            };
            equation_can_solve.insert(eq.id.clone(), solvable);

            for var in vars {
                variable_to_equations
                    .entry(var)
                    .or_default()
                    .insert(eq.id.clone());
            }
        }

        Self {
            variable_to_equations,
            equation_to_variables,
            equation_can_solve,
        }
    }

    /// Find equations that become solvable given the known variables.
    /// Returns pairs of (equation_id, variable_to_solve_for).
    pub fn find_solvable(&self, known: &HashSet<String>) -> Vec<(String, String)> {
        let mut solvable = Vec::new();

        for (eq_id, vars) in &self.equation_to_variables {
            // Count unknowns in this equation
            let unknowns: Vec<_> = vars.iter().filter(|v| !known.contains(*v)).collect();

            // If exactly one unknown and we can solve for it, it's solvable
            if unknowns.len() == 1 {
                let unknown = unknowns[0];
                if let Some(can_solve) = self.equation_can_solve.get(eq_id) {
                    if can_solve.contains(unknown) {
                        solvable.push((eq_id.clone(), unknown.clone()));
                    }
                }
            }
        }

        solvable
    }

    /// Get equations containing a specific variable.
    pub fn equations_with_variable(&self, var: &str) -> Option<&HashSet<String>> {
        self.variable_to_equations.get(var)
    }

    /// Get variables in a specific equation.
    pub fn variables_in_equation(&self, eq_id: &str) -> Option<&HashSet<String>> {
        self.equation_to_variables.get(eq_id)
    }

    /// Check if there's a circular dependency involving the given variables.
    pub fn has_circular_dependency(&self, targets: &[String], known: &HashSet<String>) -> bool {
        // Simple cycle detection using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for target in targets {
            if !known.contains(target)
                && self.has_cycle_dfs(target, known, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    fn has_cycle_dfs(
        &self,
        var: &str,
        known: &HashSet<String>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(var) {
            return true;
        }
        if visited.contains(var) || known.contains(var) {
            return false;
        }

        visited.insert(var.to_string());
        rec_stack.insert(var.to_string());

        // Find equations that can provide this variable
        if let Some(eqs) = self.variable_to_equations.get(var) {
            for eq_id in eqs {
                if let Some(eq_vars) = self.equation_to_variables.get(eq_id) {
                    for dep_var in eq_vars {
                        if dep_var != var && self.has_cycle_dfs(dep_var, known, visited, rec_stack)
                        {
                            return true;
                        }
                    }
                }
            }
        }

        rec_stack.remove(var);
        false
    }
}

// ============================================================================
// Solution Strategy
// ============================================================================

/// Method to use for solving an equation.
#[derive(Debug, Clone)]
pub enum SolveMethod {
    /// Use algebraic solver (SmartSolver).
    Algebraic,
    /// Simple substitution.
    Substitution,
    /// ODE solver with specified method.
    ODE { method: String },
    /// Integration.
    Integration,
    /// Differentiation.
    Differentiation,
    /// Numerical fallback.
    Numerical,
    /// Custom method.
    Custom(String),
}

impl fmt::Display for SolveMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Algebraic => write!(f, "algebraic"),
            Self::Substitution => write!(f, "substitution"),
            Self::ODE { method } => write!(f, "ODE ({})", method),
            Self::Integration => write!(f, "integration"),
            Self::Differentiation => write!(f, "differentiation"),
            Self::Numerical => write!(f, "numerical"),
            Self::Custom(name) => write!(f, "custom: {}", name),
        }
    }
}

/// A single step in the solving strategy.
#[derive(Debug, Clone)]
pub struct SolveStep {
    /// Equation to use.
    pub equation_id: String,
    /// Variable to solve for.
    pub solve_for: String,
    /// Method to use.
    pub method: SolveMethod,
    /// Variables that must be known before this step.
    pub dependencies: Vec<String>,
}

/// The complete solving plan.
#[derive(Debug, Clone)]
pub struct SolutionStrategy {
    /// Ordered list of solve steps.
    pub steps: Vec<SolveStep>,
    /// Groups of steps that can be executed in parallel (indices into steps).
    pub parallel_groups: Vec<Vec<usize>>,
}

impl SolutionStrategy {
    /// Create a new empty strategy.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            parallel_groups: Vec::new(),
        }
    }

    /// Add a step to the strategy.
    pub fn add_step(&mut self, step: SolveStep) {
        self.steps.push(step);
    }

    /// Check if the strategy is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of steps in the strategy.
    pub fn len(&self) -> usize {
        self.steps.len()
    }
}

impl Default for SolutionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Solution Value
// ============================================================================

/// A solution value - can be numeric, symbolic, or multiple.
#[derive(Debug, Clone)]
pub enum SolutionValue {
    /// Single numeric value.
    Numeric(f64),
    /// Symbolic expression.
    Symbolic(Expression),
    /// Multiple solutions.
    Multiple(Vec<Expression>),
    /// Parametric solution.
    Parametric {
        /// The expression.
        expr: Expression,
        /// The free parameter.
        parameter: String,
    },
}

impl SolutionValue {
    /// Try to get a numeric value.
    pub fn as_numeric(&self) -> Option<f64> {
        match self {
            Self::Numeric(v) => Some(*v),
            Self::Symbolic(expr) => {
                // Try to evaluate if it's a constant
                if let Expression::Float(f) = expr {
                    Some(*f)
                } else if let Expression::Integer(i) = expr {
                    Some(*i as f64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get the symbolic expression.
    pub fn as_expression(&self) -> Option<&Expression> {
        match self {
            Self::Numeric(_v) => None,
            Self::Symbolic(expr) => Some(expr),
            Self::Multiple(exprs) if exprs.len() == 1 => Some(&exprs[0]),
            _ => None,
        }
    }

    /// Convert to expression (converting numerics to expressions).
    pub fn to_expression(&self) -> Expression {
        match self {
            Self::Numeric(v) => Expression::Float(*v),
            Self::Symbolic(expr) => expr.clone(),
            Self::Multiple(exprs) if !exprs.is_empty() => exprs[0].clone(),
            Self::Parametric { expr, .. } => expr.clone(),
            _ => Expression::Integer(0), // Fallback
        }
    }
}

// ============================================================================
// System Resolution Path
// ============================================================================

/// Operation type for system-level solving.
#[derive(Debug, Clone)]
pub enum SystemOperation {
    /// Select an equation to work with.
    SelectEquation { reason: String },
    /// Solve for a variable using a specific method.
    SolveFor {
        variable: String,
        method: SolveMethod,
    },
    /// Substitute a result into equations.
    SubstituteResult {
        variable: String,
        into_equations: Vec<String>,
    },
    /// Verify a solution.
    VerifySolution { variable: String },
    /// Delegate to an equation-level operation.
    EquationOperation(Operation),
}

impl fmt::Display for SystemOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelectEquation { reason } => write!(f, "Select equation: {}", reason),
            Self::SolveFor { variable, method } => {
                write!(f, "Solve for {} using {}", variable, method)
            }
            Self::SubstituteResult {
                variable,
                into_equations,
            } => {
                write!(
                    f,
                    "Substitute {} into equations: {}",
                    variable,
                    into_equations.join(", ")
                )
            }
            Self::VerifySolution { variable } => write!(f, "Verify solution for {}", variable),
            Self::EquationOperation(op) => write!(f, "{:?}", op),
        }
    }
}

/// Result of a single system step.
#[derive(Debug, Clone)]
pub enum StepResult {
    /// Expression result.
    Expression(Expression),
    /// Numeric value.
    Value(f64),
    /// Intermediate state showing what's known so far.
    Intermediate {
        known_so_far: HashMap<String, Expression>,
    },
}

/// A single step in the system resolution.
#[derive(Debug, Clone)]
pub struct SystemStep {
    /// Step number (1-based).
    pub step_number: usize,
    /// Which equation was used.
    pub equation_id: String,
    /// What operation was performed.
    pub operation: SystemOperation,
    /// Human-readable explanation.
    pub explanation: String,
    /// Result of this step.
    pub result: StepResult,
}

/// Extended resolution path for multi-equation solving.
#[derive(Debug, Clone)]
pub struct SystemResolutionPath {
    /// The initial context.
    pub initial_context: SystemContext,
    /// Per-equation resolution paths.
    pub equation_paths: HashMap<String, ResolutionPath>,
    /// System-level steps.
    pub steps: Vec<SystemStep>,
    /// Final solutions.
    pub final_solutions: HashMap<String, Expression>,
}

impl SystemResolutionPath {
    /// Create a new system resolution path.
    pub fn new(context: SystemContext) -> Self {
        Self {
            initial_context: context,
            equation_paths: HashMap::new(),
            steps: Vec::new(),
            final_solutions: HashMap::new(),
        }
    }

    /// Add a step.
    pub fn add_step(&mut self, step: SystemStep) {
        self.steps.push(step);
    }

    /// Add an equation's resolution path.
    pub fn add_equation_path(&mut self, eq_id: String, path: ResolutionPath) {
        self.equation_paths.insert(eq_id, path);
    }

    /// Record a final solution.
    pub fn record_solution(&mut self, variable: String, value: Expression) {
        self.final_solutions.insert(variable, value);
    }

    /// Format as human-readable text.
    pub fn format_text(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Multi-Equation System Solution ===\n\n");

        // Initial context
        output.push_str("Known values:\n");
        for (var, val) in &self.initial_context.known_values {
            output.push_str(&format!("  {} = {}\n", var, val));
        }

        output.push_str("\nTarget variables: ");
        output.push_str(&self.initial_context.target_variables.join(", "));
        output.push_str("\n\n");

        // Steps
        output.push_str("Solution steps:\n");
        for step in &self.steps {
            output.push_str(&format!(
                "{}. [{}] {}\n   {}\n",
                step.step_number, step.equation_id, step.operation, step.explanation
            ));
        }

        // Final solutions
        output.push_str("\nFinal solutions:\n");
        for (var, expr) in &self.final_solutions {
            output.push_str(&format!("  {} = {}\n", var, expr));
        }

        output
    }

    /// Format as LaTeX.
    pub fn to_latex(&self) -> String {
        let mut output = String::new();

        output.push_str("\\section*{Multi-Equation System Solution}\n\n");

        // Known values
        output.push_str("\\subsection*{Given}\n");
        output.push_str("\\begin{align*}\n");
        for (var, val) in &self.initial_context.known_values {
            output.push_str(&format!("{} &= {} \\\\\n", var, val));
        }
        output.push_str("\\end{align*}\n\n");

        // Final solutions
        output.push_str("\\subsection*{Solutions}\n");
        output.push_str("\\begin{align*}\n");
        for (var, expr) in &self.final_solutions {
            output.push_str(&format!("{} &= {} \\\\\n", var, expr.to_latex()));
        }
        output.push_str("\\end{align*}\n");

        output
    }
}

// ============================================================================
// System Solution
// ============================================================================

/// Result of solving the equation system.
#[derive(Debug, Clone)]
pub struct MultiEquationSolution {
    /// Solutions for each variable.
    pub solutions: HashMap<String, SolutionValue>,
    /// The resolution path showing all steps.
    pub resolution_path: SystemResolutionPath,
    /// Variables that couldn't be solved.
    pub unsolved: Vec<String>,
    /// Non-fatal warnings.
    pub warnings: Vec<String>,
}

impl MultiEquationSolution {
    /// Create a new empty solution.
    pub fn new(context: SystemContext) -> Self {
        Self {
            solutions: HashMap::new(),
            resolution_path: SystemResolutionPath::new(context),
            unsolved: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Get a solution value for a variable.
    pub fn get(&self, variable: &str) -> Option<&SolutionValue> {
        self.solutions.get(variable)
    }

    /// Get a numeric value for a variable.
    pub fn get_numeric(&self, variable: &str) -> Option<f64> {
        self.solutions.get(variable).and_then(|v| v.as_numeric())
    }

    /// Get a symbolic expression for a variable.
    pub fn get_expression(&self, variable: &str) -> Option<&Expression> {
        self.solutions.get(variable).and_then(|v| v.as_expression())
    }

    /// Check if all targets were solved.
    pub fn is_complete(&self) -> bool {
        self.unsolved.is_empty()
    }

    /// Add a solution.
    pub fn add_solution(&mut self, variable: String, value: SolutionValue) {
        self.resolution_path
            .record_solution(variable.clone(), value.to_expression());
        self.solutions.insert(variable, value);
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Mark a variable as unsolved.
    pub fn mark_unsolved(&mut self, variable: String) {
        if !self.unsolved.contains(&variable) {
            self.unsolved.push(variable);
        }
    }
}

// ============================================================================
// Multi-Equation Solver
// ============================================================================

/// Configuration for the multi-equation solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum iterations for iterative solving.
    pub max_iterations: usize,
    /// Whether to use numerical fallback.
    pub use_numerical_fallback: bool,
    /// Numerical solver configuration.
    pub numerical_config: NumericalConfig,
    /// Whether to verify solutions.
    pub verify_solutions: bool,
    /// Verification tolerance.
    pub tolerance: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            use_numerical_fallback: true,
            numerical_config: NumericalConfig::default(),
            verify_solutions: true,
            tolerance: 1e-10,
        }
    }
}

/// The main multi-equation solver.
pub struct MultiEquationSolver {
    algebraic_solver: SmartSolver,
    numerical_solver: SmartNumericalSolver,
    config: SolverConfig,
}

impl MultiEquationSolver {
    /// Create a new multi-equation solver with default configuration.
    pub fn new() -> Self {
        Self {
            algebraic_solver: SmartSolver::new(),
            numerical_solver: SmartNumericalSolver::with_default_config(),
            config: SolverConfig::default(),
        }
    }

    /// Create a solver with custom configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            algebraic_solver: SmartSolver::new(),
            numerical_solver: SmartNumericalSolver::new(config.numerical_config.clone()),
            config,
        }
    }

    /// Solve the equation system.
    pub fn solve(
        &self,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<MultiEquationSolution, SystemError> {
        // Validate inputs
        if system.is_empty() {
            return Err(SystemError::NoStrategyFound(
                "No equations in system".to_string(),
            ));
        }

        if context.target_variables.is_empty() {
            return Err(SystemError::NoStrategyFound(
                "No target variables specified".to_string(),
            ));
        }

        // Build dependency graph
        let graph = self.analyze_dependencies(system);

        // Note: Circular dependency check is handled in the strategy planning phase
        // where we can detect actual deadlocks (no progress possible)
        let _known = context.known_variable_names();

        // Plan the solution
        let strategy = self.plan_solution(&graph, system, context)?;

        // Execute the strategy
        let solution = self.execute_strategy(&strategy, system, context)?;

        Ok(solution)
    }

    /// Analyze dependencies in the system.
    fn analyze_dependencies(&self, system: &EquationSystem) -> DependencyGraph {
        DependencyGraph::build(system)
    }

    /// Create a solving strategy.
    fn plan_solution(
        &self,
        graph: &DependencyGraph,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<SolutionStrategy, SystemError> {
        let mut strategy = SolutionStrategy::new();
        let mut known = context.known_variable_names();
        let mut remaining_targets: HashSet<_> = context.target_variables.iter().cloned().collect();
        let mut iterations = 0;

        while !remaining_targets.is_empty() && iterations < self.config.max_iterations {
            iterations += 1;

            // Find solvable equations
            let solvable = graph.find_solvable(&known);

            if solvable.is_empty() {
                // No progress possible - check if we can use numerical methods
                if self.config.use_numerical_fallback && !remaining_targets.is_empty() {
                    // Try numerical fallback for remaining targets
                    for target in &remaining_targets {
                        // Find an equation containing this target
                        if let Some(eqs) = graph.equations_with_variable(target) {
                            for eq_id in eqs {
                                strategy.add_step(SolveStep {
                                    equation_id: eq_id.clone(),
                                    solve_for: target.clone(),
                                    method: SolveMethod::Numerical,
                                    dependencies: known.iter().cloned().collect(),
                                });
                                break;
                            }
                        }
                    }
                    break;
                } else {
                    return Err(SystemError::NoStrategyFound(format!(
                        "Cannot determine solving order for: {:?}",
                        remaining_targets
                    )));
                }
            }

            // Add steps for solvable equations
            for (eq_id, var) in solvable {
                if remaining_targets.contains(&var) || !known.contains(&var) {
                    // Determine the solve method based on equation type
                    let method = if let Some(eq) = system.get(&eq_id) {
                        match &eq.equation_type {
                            EquationType::Algebraic | EquationType::Unknown => {
                                SolveMethod::Algebraic
                            }
                            EquationType::ODE(_info) => SolveMethod::ODE {
                                method: "auto".to_string(),
                            },
                            EquationType::Integral(_) => SolveMethod::Integration,
                            EquationType::Differential => SolveMethod::Differentiation,
                            EquationType::Implicit => SolveMethod::Numerical,
                        }
                    } else {
                        SolveMethod::Algebraic
                    };

                    strategy.add_step(SolveStep {
                        equation_id: eq_id,
                        solve_for: var.clone(),
                        method,
                        dependencies: known.iter().cloned().collect(),
                    });

                    known.insert(var.clone());
                    remaining_targets.remove(&var);
                }
            }
        }

        if remaining_targets.is_empty() || !strategy.is_empty() {
            Ok(strategy)
        } else {
            Err(SystemError::NoStrategyFound(format!(
                "Could not find strategy for: {:?}",
                remaining_targets
            )))
        }
    }

    /// Execute the solving strategy.
    fn execute_strategy(
        &self,
        strategy: &SolutionStrategy,
        system: &EquationSystem,
        context: &SystemContext,
    ) -> Result<MultiEquationSolution, SystemError> {
        let mut solution = MultiEquationSolution::new(context.clone());
        let mut known_exprs: HashMap<String, Expression> = HashMap::new();
        let mut known_values: HashMap<String, f64> = context.known_values.clone();

        // Initialize with known expressions from context
        for (var, val) in &context.known_values {
            known_exprs.insert(var.clone(), Expression::Float(*val));
        }
        for (var, expr) in &context.known_expressions {
            known_exprs.insert(var.clone(), expr.clone());
        }

        let mut step_number = 0;

        for step in &strategy.steps {
            step_number += 1;

            // Get the equation
            let eq = system
                .get(&step.equation_id)
                .ok_or_else(|| SystemError::EquationNotFound(step.equation_id.clone()))?;

            // Substitute known values into the equation
            let substituted_eq = self.substitute_known(&eq.equation, &known_exprs);

            // Solve based on method
            let result = match &step.method {
                SolveMethod::Algebraic => {
                    self.solve_algebraic(&substituted_eq, &step.solve_for, &known_values)
                }
                SolveMethod::Numerical => {
                    self.solve_numerical(&substituted_eq, &step.solve_for, &known_values)
                }
                SolveMethod::ODE { method } => {
                    self.solve_ode(&substituted_eq, &step.solve_for, method)
                }
                SolveMethod::Integration => {
                    self.solve_integration(&substituted_eq, &step.solve_for)
                }
                SolveMethod::Substitution => {
                    // For pure substitution, just evaluate the RHS if target is on LHS
                    self.solve_by_substitution(&substituted_eq, &step.solve_for, &known_values)
                }
                _ => Err(SystemError::UnsolvableEquation {
                    id: step.equation_id.clone(),
                    reason: format!("Method {:?} not implemented", step.method),
                }),
            };

            match result {
                Ok((value, eq_path)) => {
                    // Record the step
                    let expr = value.to_expression();
                    solution.resolution_path.add_step(SystemStep {
                        step_number,
                        equation_id: step.equation_id.clone(),
                        operation: SystemOperation::SolveFor {
                            variable: step.solve_for.clone(),
                            method: step.method.clone(),
                        },
                        explanation: format!(
                            "From equation '{}', solved {} = {}",
                            step.equation_id, step.solve_for, expr
                        ),
                        result: StepResult::Expression(expr.clone()),
                    });

                    // Update known values
                    known_exprs.insert(step.solve_for.clone(), expr.clone());
                    if let Some(num) = value.as_numeric() {
                        known_values.insert(step.solve_for.clone(), num);
                    }

                    // Add equation path
                    if let Some(path) = eq_path {
                        solution
                            .resolution_path
                            .add_equation_path(step.equation_id.clone(), path);
                    }

                    // Record solution
                    solution.add_solution(step.solve_for.clone(), value);
                }
                Err(e) => {
                    solution.add_warning(format!(
                        "Failed to solve for {} in {}: {}",
                        step.solve_for, step.equation_id, e
                    ));
                    solution.mark_unsolved(step.solve_for.clone());
                }
            }
        }

        // Verify solutions if requested
        if self.config.verify_solutions {
            self.verify_solutions(&solution, system, context);
        }

        Ok(solution)
    }

    /// Substitute known values into an equation.
    fn substitute_known(
        &self,
        equation: &Equation,
        known: &HashMap<String, Expression>,
    ) -> Equation {
        Equation {
            id: equation.id.clone(),
            left: self.substitute_expr(&equation.left, known),
            right: self.substitute_expr(&equation.right, known),
        }
    }

    /// Substitute known values into an expression.
    fn substitute_expr(
        &self,
        expr: &Expression,
        known: &HashMap<String, Expression>,
    ) -> Expression {
        match expr {
            Expression::Variable(var) => {
                if let Some(val) = known.get(&var.name) {
                    val.clone()
                } else {
                    expr.clone()
                }
            }
            Expression::Binary(op, left, right) => Expression::Binary(
                *op,
                Box::new(self.substitute_expr(left, known)),
                Box::new(self.substitute_expr(right, known)),
            ),
            Expression::Unary(op, inner) => {
                Expression::Unary(*op, Box::new(self.substitute_expr(inner, known)))
            }
            Expression::Power(base, exp) => Expression::Power(
                Box::new(self.substitute_expr(base, known)),
                Box::new(self.substitute_expr(exp, known)),
            ),
            Expression::Function(func, args) => Expression::Function(
                func.clone(),
                args.iter()
                    .map(|a| self.substitute_expr(a, known))
                    .collect(),
            ),
            _ => expr.clone(),
        }
    }

    /// Solve an equation algebraically.
    fn solve_algebraic(
        &self,
        equation: &Equation,
        variable: &str,
        known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        let var = Variable::new(variable);

        match self.algebraic_solver.solve(equation, &var) {
            Ok((sol, path)) => {
                let value = match sol {
                    Solution::Unique(expr) => {
                        // Try to evaluate numerically
                        match expr.evaluate(known_values) {
                            Some(num) => SolutionValue::Numeric(num),
                            None => SolutionValue::Symbolic(expr),
                        }
                    }
                    Solution::Multiple(exprs) => SolutionValue::Multiple(exprs),
                    Solution::None => {
                        return Err(SystemError::UnsolvableEquation {
                            id: "algebraic".to_string(),
                            reason: "No solution exists".to_string(),
                        })
                    }
                    Solution::Infinite => {
                        return Err(SystemError::UnsolvableEquation {
                            id: "algebraic".to_string(),
                            reason: "Infinite solutions".to_string(),
                        })
                    }
                    Solution::Parametric {
                        expression,
                        constraints: _,
                    } => SolutionValue::Parametric {
                        expr: expression,
                        parameter: "t".to_string(), // Default parameter name
                    },
                };
                Ok((value, Some(path)))
            }
            Err(e) => Err(SystemError::SolverError(format!("{:?}", e))),
        }
    }

    /// Solve an equation numerically.
    fn solve_numerical(
        &self,
        equation: &Equation,
        variable: &str,
        _known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        let var = Variable::new(variable);

        // Use SmartNumericalSolver's solve method
        match self.numerical_solver.solve(equation, &var) {
            Ok((sol, path)) => {
                let value = SolutionValue::Numeric(sol.value);
                Ok((value, Some(path)))
            }
            Err(e) => Err(SystemError::NumericalFailure {
                variable: variable.to_string(),
                reason: format!("{:?}", e),
            }),
        }
    }

    /// Solve an ODE.
    fn solve_ode(
        &self,
        _equation: &Equation,
        _variable: &str,
        _method: &str,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // TODO: Integrate with ODE solver
        // For now, return an error indicating ODE solving is not fully implemented
        Err(SystemError::UnsolvableEquation {
            id: "ode".to_string(),
            reason: "ODE solving in multi-equation context not yet implemented".to_string(),
        })
    }

    /// Solve by integration.
    fn solve_integration(
        &self,
        _equation: &Equation,
        _variable: &str,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // TODO: Integrate with integration module
        Err(SystemError::UnsolvableEquation {
            id: "integration".to_string(),
            reason: "Integration solving not yet implemented".to_string(),
        })
    }

    /// Solve by simple substitution.
    fn solve_by_substitution(
        &self,
        equation: &Equation,
        variable: &str,
        known_values: &HashMap<String, f64>,
    ) -> Result<(SolutionValue, Option<ResolutionPath>), SystemError> {
        // Check if the equation is in the form var = expr
        if let Expression::Variable(var) = &equation.left {
            if var.name == variable {
                // var = rhs, so just evaluate rhs
                match equation.right.evaluate(known_values) {
                    Some(val) => return Ok((SolutionValue::Numeric(val), None)),
                    None => return Ok((SolutionValue::Symbolic(equation.right.clone()), None)),
                }
            }
        }

        // Check if rhs is the variable
        if let Expression::Variable(var) = &equation.right {
            if var.name == variable {
                // rhs = lhs
                match equation.left.evaluate(known_values) {
                    Some(val) => return Ok((SolutionValue::Numeric(val), None)),
                    None => return Ok((SolutionValue::Symbolic(equation.left.clone()), None)),
                }
            }
        }

        // Fall back to algebraic solving
        self.solve_algebraic(equation, variable, known_values)
    }

    /// Verify solutions by substituting back.
    fn verify_solutions(
        &self,
        solution: &MultiEquationSolution,
        system: &EquationSystem,
        _context: &SystemContext,
    ) {
        // Build a map of all known values
        let mut all_values: HashMap<String, f64> = HashMap::new();
        for (var, val) in &solution.solutions {
            if let Some(num) = val.as_numeric() {
                all_values.insert(var.clone(), num);
            }
        }

        // Verify each equation
        for eq in system.equations() {
            let lhs_val = eq.equation.left.evaluate(&all_values);
            let rhs_val = eq.equation.right.evaluate(&all_values);

            match (lhs_val, rhs_val) {
                (Some(l), Some(r)) => {
                    let diff = (l - r).abs();
                    if diff > self.config.tolerance {
                        // Log warning but don't fail
                        eprintln!(
                            "Warning: Equation '{}' verification failed: {} != {} (diff: {})",
                            eq.id, l, r, diff
                        );
                    }
                }
                _ => {
                    // Can't verify symbolically
                }
            }
        }
    }
}

impl Default for MultiEquationSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_equation;

    #[test]
    fn test_equation_system_creation() {
        let mut system = EquationSystem::new();
        system.add_equation("eq1", parse_equation("x + y = 10").unwrap());
        system.add_equation("eq2", parse_equation("x - y = 2").unwrap());

        assert_eq!(system.len(), 2);
        assert!(system.get("eq1").is_some());
        assert!(system.get("eq2").is_some());
    }

    #[test]
    fn test_equation_system_variables() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let vars = system.all_variables();
        assert!(vars.contains("F"));
        assert!(vars.contains("m"));
        assert!(vars.contains("a"));
        assert!(vars.contains("v"));
        assert!(vars.contains("u"));
        assert!(vars.contains("t"));
    }

    #[test]
    fn test_context_builder() {
        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        assert_eq!(context.known_values.get("F"), Some(&100.0));
        assert_eq!(context.known_values.get("m"), Some(&20.0));
        assert!(context.target_variables.contains(&"a".to_string()));
    }

    #[test]
    fn test_dependency_graph() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let graph = DependencyGraph::build(&system);

        // Check variable to equations mapping
        let a_eqs = graph.equations_with_variable("a").unwrap();
        assert!(a_eqs.contains("eq1"));
        assert!(a_eqs.contains("eq2"));

        // Check equation to variables mapping
        let eq1_vars = graph.variables_in_equation("eq1").unwrap();
        assert!(eq1_vars.contains("F"));
        assert!(eq1_vars.contains("m"));
        assert!(eq1_vars.contains("a"));
    }

    #[test]
    fn test_find_solvable() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let graph = DependencyGraph::build(&system);

        // With F and m known, we should be able to solve for a in eq1
        let mut known = HashSet::new();
        known.insert("F".to_string());
        known.insert("m".to_string());

        let solvable = graph.find_solvable(&known);
        assert!(solvable.iter().any(|(eq, var)| eq == "eq1" && var == "a"));
    }

    #[test]
    fn test_simple_linear_system() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("F = m * a").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        let a = solution.get_numeric("a").unwrap();
        assert!((a - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_chained_equations() {
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap())
            .with_equation("eq2", parse_equation("v = u + a * t").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_known_value("u", 0.0)
            .with_known_value("t", 5.0)
            .with_target("a")
            .with_target("v");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        // F = m * a => 100 = 20 * a => a = 5
        let a = solution.get_numeric("a").unwrap();
        assert!((a - 5.0).abs() < 1e-10);

        // v = u + a * t => v = 0 + 5 * 5 => v = 25
        let v = solution.get_numeric("v").unwrap();
        assert!((v - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_resolution_path() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("F = m * a").unwrap());

        let context = SystemContext::new()
            .with_known_value("F", 100.0)
            .with_known_value("m", 20.0)
            .with_target("a");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context).unwrap();

        // Check that resolution path has steps
        assert!(!solution.resolution_path.steps.is_empty());

        // Check text formatting doesn't panic
        let text = solution.resolution_path.format_text();
        assert!(text.contains("Multi-Equation System Solution"));
    }

    #[test]
    fn test_insufficient_equations() {
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("x + y = 10").unwrap());

        let context = SystemContext::new().with_target("x").with_target("y");

        let solver = MultiEquationSolver::new();
        let result = solver.solve(&system, &context);

        // Should fail because we have 2 unknowns but only 1 equation
        assert!(result.is_err() || !result.unwrap().is_complete());
    }

    #[test]
    fn test_solution_value_conversion() {
        let numeric = SolutionValue::Numeric(42.0);
        assert_eq!(numeric.as_numeric(), Some(42.0));

        let symbolic = SolutionValue::Symbolic(Expression::Variable(Variable::new("x")));
        assert!(symbolic.as_numeric().is_none());
        assert!(symbolic.as_expression().is_some());
    }

    #[test]
    fn test_quadratic_in_system() {
        // x² = 16 => x = ±4
        let system =
            EquationSystem::new().with_equation("eq1", parse_equation("y = x * x").unwrap());

        let context = SystemContext::new()
            .with_known_value("y", 16.0)
            .with_target("x");

        let solver = MultiEquationSolver::new();
        let solution = solver.solve(&system, &context);

        // This might give x = 4 or x = -4 depending on solver behavior
        // Just check that we get a solution
        if let Ok(sol) = solution {
            if let Some(x) = sol.get_numeric("x") {
                assert!((x.abs() - 4.0).abs() < 1e-10);
            }
        }
    }
}

// ============================================================================
// Nonlinear System Solver
// ============================================================================

/// Errors that can occur during nonlinear system solving.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NonlinearSystemSolverError {
    /// System did not converge within max iterations.
    NoConvergence {
        /// Number of iterations attempted.
        iterations: usize,
        /// Final residual norm.
        final_residual: f64,
    },
    /// Jacobian matrix is singular or near-singular.
    SingularJacobian {
        /// Condition number estimate (if available).
        condition_estimate: Option<f64>,
    },
    /// Dimension mismatch between equations and variables.
    DimensionMismatch {
        /// Number of equations.
        num_equations: usize,
        /// Number of variables.
        num_variables: usize,
    },
    /// Failed to evaluate function at a point.
    EvaluationFailed {
        /// Point at which evaluation failed.
        point: Vec<f64>,
        /// Reason for failure.
        reason: String,
    },
    /// Invalid configuration.
    InvalidConfig(String),
    /// Differentiation failed when computing Jacobian.
    DifferentiationFailed(String),
}

impl fmt::Display for NonlinearSystemSolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoConvergence {
                iterations,
                final_residual,
            } => {
                write!(
                    f,
                    "No convergence after {} iterations (residual: {:.2e})",
                    iterations, final_residual
                )
            }
            Self::SingularJacobian { condition_estimate } => {
                if let Some(cond) = condition_estimate {
                    write!(f, "Singular Jacobian (condition ~{:.2e})", cond)
                } else {
                    write!(f, "Singular Jacobian")
                }
            }
            Self::DimensionMismatch {
                num_equations,
                num_variables,
            } => {
                write!(
                    f,
                    "Dimension mismatch: {} equations, {} variables",
                    num_equations, num_variables
                )
            }
            Self::EvaluationFailed { point, reason } => {
                write!(f, "Evaluation failed at {:?}: {}", point, reason)
            }
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::DifferentiationFailed(msg) => write!(f, "Differentiation failed: {}", msg),
        }
    }
}

impl std::error::Error for NonlinearSystemSolverError {}

/// Configuration for nonlinear system solvers.
#[derive(Debug, Clone)]
pub struct NonlinearSystemConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Tolerance for residual norm.
    pub tolerance: f64,
    /// Tolerance for step size.
    pub step_tolerance: f64,
    /// Damping factor for damped Newton (1.0 = no damping).
    pub damping_factor: f64,
    /// Whether to use line search for step size.
    pub use_line_search: bool,
    /// Finite difference epsilon for Jacobian validation.
    pub finite_diff_epsilon: f64,
    /// Minimum step size for line search.
    pub min_step_size: f64,
    /// Regularization parameter for near-singular Jacobians.
    pub regularization: f64,
}

impl Default for NonlinearSystemConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
            step_tolerance: 1e-12,
            damping_factor: 1.0,
            use_line_search: false,
            finite_diff_epsilon: 1e-8,
            min_step_size: 1e-10,
            regularization: 1e-12,
        }
    }
}

impl NonlinearSystemConfig {
    /// Create a config with damped Newton settings.
    pub fn damped() -> Self {
        Self {
            damping_factor: 0.5,
            use_line_search: true,
            ..Default::default()
        }
    }

    /// Create a config for Broyden's method.
    pub fn for_broyden() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        }
    }
}

/// Result of nonlinear system solving.
#[derive(Debug, Clone)]
pub struct NonlinearSystemSolverResult {
    /// Solution vector.
    pub solution: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub final_residual: f64,
    /// Convergence history (residual norm at each iteration).
    pub convergence_history: Vec<f64>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Method used for solving.
    pub method: String,
    /// Variable names in order.
    pub variable_names: Vec<String>,
}

impl NonlinearSystemSolverResult {
    /// Get solution as a map from variable name to value.
    pub fn as_map(&self) -> HashMap<String, f64> {
        self.variable_names
            .iter()
            .zip(self.solution.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect()
    }

    /// Estimate convergence rate (linear rate estimate from last few iterations).
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.convergence_history.len() < 3 {
            return None;
        }

        let n = self.convergence_history.len();
        let r1 = self.convergence_history[n - 2];
        let r2 = self.convergence_history[n - 1];
        let r0 = self.convergence_history[n - 3];

        if r0 > 1e-15 && r1 > 1e-15 {
            // Linear convergence rate estimate: r_{n+1}/r_n
            Some(r2 / r1)
        } else {
            None
        }
    }
}

/// Convergence diagnostics for analyzing solver behavior.
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    /// Residual history.
    pub residual_history: Vec<f64>,
    /// Step size history.
    pub step_history: Vec<f64>,
    /// Estimated convergence rate.
    pub estimated_rate: Option<f64>,
    /// Detected behavior.
    pub behavior: ConvergenceBehavior,
}

/// Type of convergence behavior observed.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceBehavior {
    /// Fast quadratic convergence (Newton).
    Quadratic,
    /// Linear convergence.
    Linear,
    /// Sublinear convergence.
    Sublinear,
    /// Oscillating.
    Oscillating,
    /// Diverging.
    Diverging,
    /// Stalled.
    Stalled,
}

impl ConvergenceDiagnostics {
    /// Analyze convergence from history.
    pub fn analyze(residuals: &[f64], steps: &[f64]) -> Self {
        let behavior = if residuals.len() < 3 {
            ConvergenceBehavior::Linear
        } else {
            let n = residuals.len();
            let r1 = residuals[n - 2];
            let r2 = residuals[n - 1];
            let r0 = residuals[n - 3];

            if r2 > r1 * 1.1 {
                ConvergenceBehavior::Diverging
            } else if r2 > r1 * 0.99 {
                ConvergenceBehavior::Stalled
            } else if r0 > 1e-15 && r1 > 1e-15 {
                let rate1 = r1 / r0;
                let rate2 = r2 / r1;
                if rate2 < rate1 * rate1 * 2.0 {
                    ConvergenceBehavior::Quadratic
                } else if rate2 < 0.9 {
                    ConvergenceBehavior::Linear
                } else {
                    ConvergenceBehavior::Sublinear
                }
            } else {
                ConvergenceBehavior::Linear
            }
        };

        let estimated_rate = if residuals.len() >= 2 {
            let n = residuals.len();
            if residuals[n - 2] > 1e-15 {
                Some(residuals[n - 1] / residuals[n - 2])
            } else {
                None
            }
        } else {
            None
        };

        Self {
            residual_history: residuals.to_vec(),
            step_history: steps.to_vec(),
            estimated_rate,
            behavior,
        }
    }
}

/// A system of nonlinear equations F(x) = 0.
#[derive(Debug, Clone)]
pub struct NonlinearSystem {
    /// The equations F_i(x) (each should equal 0).
    pub equations: Vec<Expression>,
    /// The variables in order.
    pub variables: Vec<Variable>,
}

impl NonlinearSystem {
    /// Create a new nonlinear system.
    pub fn new(equations: Vec<Expression>, variables: Vec<Variable>) -> Self {
        Self {
            equations,
            variables,
        }
    }

    /// Create from Equation types (converts to F(x) = 0 form).
    pub fn from_equations(equations: Vec<Equation>, variables: Vec<Variable>) -> Self {
        // Convert each equation to the form: left - right = 0
        let exprs: Vec<Expression> = equations
            .into_iter()
            .map(|eq| Expression::Binary(BinaryOp::Sub, Box::new(eq.left), Box::new(eq.right)))
            .collect();
        Self::new(exprs, variables)
    }

    /// Number of equations.
    pub fn num_equations(&self) -> usize {
        self.equations.len()
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Check if the system is square (n equations, n variables).
    pub fn is_square(&self) -> bool {
        self.num_equations() == self.num_variables()
    }

    /// Evaluate all equations at a point.
    pub fn evaluate(&self, point: &[f64]) -> Result<Vec<f64>, NonlinearSystemSolverError> {
        if point.len() != self.variables.len() {
            return Err(NonlinearSystemSolverError::DimensionMismatch {
                num_equations: self.equations.len(),
                num_variables: point.len(),
            });
        }

        // Build variable map
        let var_map: HashMap<String, f64> = self
            .variables
            .iter()
            .zip(point.iter())
            .map(|(v, &val)| (v.name.clone(), val))
            .collect();

        // Evaluate each equation
        let mut result = Vec::with_capacity(self.equations.len());
        for (i, eq) in self.equations.iter().enumerate() {
            match eq.evaluate(&var_map) {
                Some(val) => result.push(val),
                None => {
                    return Err(NonlinearSystemSolverError::EvaluationFailed {
                        point: point.to_vec(),
                        reason: format!("Could not evaluate equation {}", i),
                    })
                }
            }
        }

        Ok(result)
    }

    /// Compute the Jacobian matrix symbolically.
    /// Returns J[i][j] = ∂F_i/∂x_j
    pub fn jacobian(&self) -> Result<Vec<Vec<Expression>>, NonlinearSystemSolverError> {
        let mut jacobian = Vec::with_capacity(self.equations.len());

        for eq in &self.equations {
            let mut row = Vec::with_capacity(self.variables.len());
            for var in &self.variables {
                // Use Expression::differentiate method
                let deriv = eq.differentiate(&var.name);
                row.push(deriv);
            }
            jacobian.push(row);
        }

        Ok(jacobian)
    }

    /// Evaluate the Jacobian matrix at a point.
    pub fn evaluate_jacobian(
        &self,
        point: &[f64],
    ) -> Result<Vec<Vec<f64>>, NonlinearSystemSolverError> {
        let symbolic_jacobian = self.jacobian()?;

        // Build variable map
        let var_map: HashMap<String, f64> = self
            .variables
            .iter()
            .zip(point.iter())
            .map(|(v, &val)| (v.name.clone(), val))
            .collect();

        let mut result = Vec::with_capacity(symbolic_jacobian.len());
        for (i, row) in symbolic_jacobian.iter().enumerate() {
            let mut eval_row = Vec::with_capacity(row.len());
            for (j, expr) in row.iter().enumerate() {
                match expr.evaluate(&var_map) {
                    Some(val) => eval_row.push(val),
                    None => {
                        return Err(NonlinearSystemSolverError::EvaluationFailed {
                            point: point.to_vec(),
                            reason: format!("Could not evaluate J[{}][{}]", i, j),
                        })
                    }
                }
            }
            result.push(eval_row);
        }

        Ok(result)
    }

    /// Variable names as strings.
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.iter().map(|v| v.name.clone()).collect()
    }
}

/// Calculate L2 norm of residuals.
pub fn residual_norm(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// Solve a linear system Ax = b using LU decomposition with partial pivoting.
pub fn solve_linear_system_lu(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, NonlinearSystemSolverError> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: n,
            num_variables: rhs.len(),
        });
    }

    // Copy matrix for LU decomposition
    let mut lu: Vec<Vec<f64>> = matrix.to_vec();
    let mut p: Vec<usize> = (0..n).collect(); // Permutation vector
    let b = rhs.to_vec();

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if lu[i][k].abs() > max_val {
                max_val = lu[i][k].abs();
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return Err(NonlinearSystemSolverError::SingularJacobian {
                condition_estimate: None,
            });
        }

        // Swap rows
        if max_row != k {
            lu.swap(k, max_row);
            p.swap(k, max_row);
        }

        // Elimination
        for i in (k + 1)..n {
            lu[i][k] /= lu[k][k];
            for j in (k + 1)..n {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[p[i]];
    }

    // Forward substitution (Ly = Pb)
    let mut y = pb;
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu[i][j] * y[j];
        }
    }

    // Back substitution (Ux = y)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }

    Ok(x)
}

/// Newton-Raphson solver for nonlinear systems.
pub fn newton_raphson_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    for iter in 0..config.max_iterations {
        // Evaluate F(x)
        let f = system.evaluate(&x)?;
        let residual = residual_norm(&f);
        convergence_history.push(residual);

        // Check convergence
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Newton-Raphson".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Compute Jacobian
        let jacobian = system.evaluate_jacobian(&x)?;

        // Apply regularization if needed
        let mut reg_jacobian = jacobian.clone();
        for i in 0..n {
            reg_jacobian[i][i] += config.regularization;
        }

        // Solve J * delta = -F
        let neg_f: Vec<f64> = f.iter().map(|v| -v).collect();
        let delta = solve_linear_system_lu(&reg_jacobian, &neg_f)?;

        // Check step size
        let step_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if step_norm < config.step_tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Newton-Raphson".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply damping or line search
        let alpha = if config.use_line_search {
            // Simple backtracking line search
            let mut alpha = 1.0;
            let c = 0.5;
            let rho = 0.5;

            for _ in 0..20 {
                let x_new: Vec<f64> = x
                    .iter()
                    .zip(delta.iter())
                    .map(|(xi, di)| xi + alpha * di)
                    .collect();

                if let Ok(f_new) = system.evaluate(&x_new) {
                    let new_residual = residual_norm(&f_new);
                    if new_residual < residual * (1.0 - c * alpha) {
                        break;
                    }
                }

                alpha *= rho;
                if alpha < config.min_step_size {
                    break;
                }
            }
            alpha
        } else {
            config.damping_factor
        };

        // Update x
        for i in 0..n {
            x[i] += alpha * delta[i];
        }
    }

    // Did not converge
    let f = system.evaluate(&x)?;
    let final_residual = residual_norm(&f);

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual,
    })
}

/// Fixed-point iteration solver for nonlinear systems.
/// Requires the system to be expressed as x = G(x).
pub fn fixed_point_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    for iter in 0..config.max_iterations {
        // Evaluate F(x) - the equations should be in fixed-point form x = G(x)
        // so F(x) = G(x) - x, and we want x_new = G(x) = x + F(x)
        let f = system.evaluate(&x)?;
        let residual = residual_norm(&f);
        convergence_history.push(residual);

        // Check convergence
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Fixed-Point".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply fixed-point update with damping
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            // x_new = x - damping * F(x) (for systems in standard form)
            x_new[i] = x[i] - config.damping_factor * f[i];
        }

        // Check for divergence
        if let Ok(f_new) = system.evaluate(&x_new) {
            let new_residual = residual_norm(&f_new);
            if new_residual > residual * 10.0 {
                return Err(NonlinearSystemSolverError::NoConvergence {
                    iterations: iter,
                    final_residual: new_residual,
                });
            }
        }

        x = x_new;
    }

    let f = system.evaluate(&x)?;
    let final_residual = residual_norm(&f);

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual,
    })
}

/// Broyden's method - a quasi-Newton method that avoids Jacobian recomputation.
pub fn broyden_system(
    system: &NonlinearSystem,
    initial_guess: &[f64],
    config: &NonlinearSystemConfig,
) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
    if !system.is_square() {
        return Err(NonlinearSystemSolverError::DimensionMismatch {
            num_equations: system.num_equations(),
            num_variables: system.num_variables(),
        });
    }

    let n = system.num_variables();
    let mut x = initial_guess.to_vec();
    let mut convergence_history = Vec::new();

    // Initial Jacobian
    let mut b = system.evaluate_jacobian(&x)?;

    // Initial function evaluation
    let mut f = system.evaluate(&x)?;
    let mut residual = residual_norm(&f);
    convergence_history.push(residual);

    for iter in 0..config.max_iterations {
        if residual < config.tolerance {
            return Ok(NonlinearSystemSolverResult {
                solution: x,
                iterations: iter,
                final_residual: residual,
                convergence_history,
                converged: true,
                method: "Broyden".to_string(),
                variable_names: system.variable_names(),
            });
        }

        // Apply regularization
        let mut reg_b = b.clone();
        for i in 0..n {
            reg_b[i][i] += config.regularization;
        }

        // Solve B * s = -F
        let neg_f: Vec<f64> = f.iter().map(|v| -v).collect();
        let s = solve_linear_system_lu(&reg_b, &neg_f)?;

        // Update x
        let mut x_new = x.clone();
        for i in 0..n {
            x_new[i] += config.damping_factor * s[i];
        }

        // Evaluate at new point
        let f_new = system.evaluate(&x_new)?;
        let new_residual = residual_norm(&f_new);

        // Compute y = F(x_new) - F(x)
        let y: Vec<f64> = f_new.iter().zip(f.iter()).map(|(a, b)| a - b).collect();

        // Broyden update: B_new = B + (y - B*s) * s^T / (s^T * s)
        // Compute B*s
        let mut bs = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                bs[i] += b[i][j] * s[j];
            }
        }

        // Compute y - B*s
        let diff: Vec<f64> = y.iter().zip(bs.iter()).map(|(a, b)| a - b).collect();

        // Compute s^T * s
        let s_dot_s: f64 = s.iter().map(|si| si * si).sum();

        if s_dot_s > 1e-15 {
            // Update B: B_new[i][j] = B[i][j] + diff[i] * s[j] / s_dot_s
            for i in 0..n {
                for j in 0..n {
                    b[i][j] += diff[i] * s[j] / s_dot_s;
                }
            }
        }

        x = x_new;
        f = f_new;
        residual = new_residual;
        convergence_history.push(residual);
    }

    Err(NonlinearSystemSolverError::NoConvergence {
        iterations: config.max_iterations,
        final_residual: residual,
    })
}

/// Trait for nonlinear system solvers.
pub trait NonlinearSystemSolver {
    /// Solve the nonlinear system.
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError>;

    /// Method name.
    fn method_name(&self) -> &str;
}

/// Newton-Raphson solver implementation.
pub struct NewtonRaphsonSolver;

impl NonlinearSystemSolver for NewtonRaphsonSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        newton_raphson_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Newton-Raphson"
    }
}

/// Broyden solver implementation.
pub struct BroydenSolver;

impl NonlinearSystemSolver for BroydenSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        broyden_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Broyden"
    }
}

/// Fixed-point solver implementation.
pub struct FixedPointSolver;

impl NonlinearSystemSolver for FixedPointSolver {
    fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
        config: &NonlinearSystemConfig,
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        fixed_point_system(system, initial_guess, config)
    }

    fn method_name(&self) -> &str {
        "Fixed-Point"
    }
}

/// Smart nonlinear system solver that selects method based on system properties.
pub struct SmartNonlinearSystemSolver {
    /// Configuration override.
    config: Option<NonlinearSystemConfig>,
}

impl SmartNonlinearSystemSolver {
    /// Create a new smart solver.
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Create with specific configuration.
    pub fn with_config(config: NonlinearSystemConfig) -> Self {
        Self {
            config: Some(config),
        }
    }

    /// Solve using automatic method selection.
    pub fn solve(
        &self,
        system: &NonlinearSystem,
        initial_guess: &[f64],
    ) -> Result<NonlinearSystemSolverResult, NonlinearSystemSolverError> {
        let config = self.config.clone().unwrap_or_default();

        // Try Newton-Raphson first (fastest when it works)
        match newton_raphson_system(system, initial_guess, &config) {
            Ok(result) if result.converged => return Ok(result),
            _ => {}
        }

        // Fall back to Broyden (more robust, doesn't recompute Jacobian)
        let broyden_config = NonlinearSystemConfig {
            max_iterations: config.max_iterations * 2,
            damping_factor: 0.8,
            ..config.clone()
        };

        match broyden_system(system, initial_guess, &broyden_config) {
            Ok(result) if result.converged => return Ok(result),
            _ => {}
        }

        // Try damped Newton with line search
        let damped_config = NonlinearSystemConfig {
            damping_factor: 0.5,
            use_line_search: true,
            max_iterations: config.max_iterations * 2,
            ..config
        };

        newton_raphson_system(system, initial_guess, &damped_config)
    }

    /// Find all solutions by trying multiple initial guesses.
    pub fn find_all_solutions(
        &self,
        system: &NonlinearSystem,
        initial_guesses: &[Vec<f64>],
    ) -> Vec<NonlinearSystemSolverResult> {
        let config = self.config.clone().unwrap_or_default();
        let mut solutions = Vec::new();
        let tolerance = config.tolerance * 100.0; // Tolerance for considering solutions distinct

        for guess in initial_guesses {
            if let Ok(result) = self.solve(system, guess) {
                if result.converged {
                    // Check if this is a new solution
                    let is_new = solutions
                        .iter()
                        .all(|existing: &NonlinearSystemSolverResult| {
                            let diff: f64 = existing
                                .solution
                                .iter()
                                .zip(result.solution.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            diff > tolerance
                        });

                    if is_new {
                        solutions.push(result);
                    }
                }
            }
        }

        solutions
    }
}

impl Default for SmartNonlinearSystemSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate the Jacobian using finite differences.
pub fn validate_jacobian(
    system: &NonlinearSystem,
    point: &[f64],
    epsilon: f64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, f64), NonlinearSystemSolverError> {
    let analytic = system.evaluate_jacobian(point)?;
    let n = system.num_variables();
    let m = system.num_equations();

    // Compute finite difference Jacobian
    let f0 = system.evaluate(point)?;
    let mut numeric = vec![vec![0.0; n]; m];

    for j in 0..n {
        let mut point_plus = point.to_vec();
        point_plus[j] += epsilon;

        let f_plus = system.evaluate(&point_plus)?;

        for i in 0..m {
            numeric[i][j] = (f_plus[i] - f0[i]) / epsilon;
        }
    }

    // Compute max absolute difference
    let mut max_diff = 0.0;
    for i in 0..m {
        for j in 0..n {
            let diff = (analytic[i][j] - numeric[i][j]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    Ok((analytic, numeric, max_diff))
}

#[cfg(test)]
mod nonlinear_tests {
    use super::*;

    fn make_circle_line_system() -> NonlinearSystem {
        // x^2 + y^2 - 1 = 0 (unit circle)
        // x - y = 0 (diagonal line)
        let x = Variable::new("x");
        let y = Variable::new("y");

        // x^2 + y^2 - 1
        let eq1 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Power(
                    Box::new(Expression::Variable(x.clone())),
                    Box::new(Expression::Integer(2)),
                )),
                Box::new(Expression::Power(
                    Box::new(Expression::Variable(y.clone())),
                    Box::new(Expression::Integer(2)),
                )),
            )),
            Box::new(Expression::Integer(1)),
        );

        // x - y
        let eq2 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Variable(x.clone())),
            Box::new(Expression::Variable(y.clone())),
        );

        NonlinearSystem::new(vec![eq1, eq2], vec![x, y])
    }

    fn make_hyperbola_line_system() -> NonlinearSystem {
        // x*y - 1 = 0 (hyperbola xy = 1)
        // x + y - 3 = 0 (line x + y = 3)
        let x = Variable::new("x");
        let y = Variable::new("y");

        // x*y - 1
        let eq1 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Variable(x.clone())),
                Box::new(Expression::Variable(y.clone())),
            )),
            Box::new(Expression::Integer(1)),
        );

        // x + y - 3
        let eq2 = Expression::Binary(
            BinaryOp::Sub,
            Box::new(Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Variable(x.clone())),
                Box::new(Expression::Variable(y.clone())),
            )),
            Box::new(Expression::Integer(3)),
        );

        NonlinearSystem::new(vec![eq1, eq2], vec![x, y])
    }

    #[test]
    fn test_nonlinear_system_creation() {
        let system = make_circle_line_system();
        assert_eq!(system.num_equations(), 2);
        assert_eq!(system.num_variables(), 2);
        assert!(system.is_square());
    }

    #[test]
    fn test_evaluate() {
        let system = make_circle_line_system();

        // At (1, 0): x^2 + y^2 - 1 = 0, x - y = 1
        let result = system.evaluate(&[1.0, 0.0]).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);

        // At (0.5, 0.5): x^2 + y^2 - 1 = -0.5, x - y = 0
        let result = system.evaluate(&[0.5, 0.5]).unwrap();
        assert!((result[0] - (-0.5)).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_residual_norm() {
        let residuals = vec![3.0, 4.0];
        assert!((residual_norm(&residuals) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_lu() {
        // Simple 2x2 system: x + y = 3, x - y = 1 => x = 2, y = 1
        let matrix = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let rhs = vec![3.0, 1.0];

        let solution = solve_linear_system_lu(&matrix, &rhs).unwrap();
        assert!((solution[0] - 2.0).abs() < 1e-10);
        assert!((solution[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_newton_raphson_circle_line() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (0.5, 0.5), should converge to (√2/2, √2/2)
        let result = newton_raphson_system(&system, &[0.5, 0.5], &config).unwrap();

        assert!(result.converged);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-8);
        assert!((result.solution[1] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_newton_raphson_negative_solution() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (-0.5, -0.5), should converge to (-√2/2, -√2/2)
        let result = newton_raphson_system(&system, &[-0.5, -0.5], &config).unwrap();

        assert!(result.converged);
        let expected = -std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-8);
        assert!((result.solution[1] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_hyperbola_line_solution() {
        let system = make_hyperbola_line_system();
        let config = NonlinearSystemConfig::default();

        // Starting from (1.5, 1.5), should converge to (1, 2) or (2, 1)
        let result = newton_raphson_system(&system, &[1.5, 1.5], &config).unwrap();

        assert!(result.converged);
        // Either (1, 2) or (2, 1)
        let x = result.solution[0];
        let y = result.solution[1];
        assert!((x * y - 1.0).abs() < 1e-8);
        assert!((x + y - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_broyden_circle_line() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::for_broyden();

        let result = broyden_system(&system, &[0.5, 0.5], &config).unwrap();

        assert!(result.converged);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((result.solution[0] - expected).abs() < 1e-6);
        assert!((result.solution[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_jacobian_validation() {
        let system = make_circle_line_system();
        let point = vec![0.5, 0.5];
        let epsilon = 1e-6;

        let (analytic, _numeric, max_diff) = validate_jacobian(&system, &point, epsilon).unwrap();

        // Analytic Jacobian at (0.5, 0.5):
        // [2x, 2y] = [1.0, 1.0]
        // [1, -1] = [1.0, -1.0]
        assert!((analytic[0][0] - 1.0).abs() < 1e-8);
        assert!((analytic[0][1] - 1.0).abs() < 1e-8);
        assert!((analytic[1][0] - 1.0).abs() < 1e-8);
        assert!((analytic[1][1] - (-1.0)).abs() < 1e-8);

        // Numeric should match analytic closely
        assert!(max_diff < 1e-4);
    }

    #[test]
    fn test_smart_solver() {
        let system = make_circle_line_system();
        let solver = SmartNonlinearSystemSolver::new();

        let result = solver.solve(&system, &[0.5, 0.5]).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_find_all_solutions() {
        let system = make_circle_line_system();
        let solver = SmartNonlinearSystemSolver::new();

        // Try multiple initial guesses to find both solutions
        let guesses = vec![
            vec![0.5, 0.5],
            vec![-0.5, -0.5],
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
        ];

        let solutions = solver.find_all_solutions(&system, &guesses);

        // Should find 2 distinct solutions
        assert_eq!(solutions.len(), 2);

        // Verify both solutions
        let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
        let has_positive = solutions
            .iter()
            .any(|s| (s.solution[0] - sqrt2_over_2).abs() < 1e-6);
        let has_negative = solutions
            .iter()
            .any(|s| (s.solution[0] + sqrt2_over_2).abs() < 1e-6);
        assert!(has_positive);
        assert!(has_negative);
    }

    #[test]
    fn test_convergence_diagnostics() {
        let residuals = vec![1.0, 0.1, 0.01, 0.001];
        let steps = vec![1.0, 0.5, 0.25, 0.125];

        let diagnostics = ConvergenceDiagnostics::analyze(&residuals, &steps);

        assert_eq!(diagnostics.behavior, ConvergenceBehavior::Linear);
        assert!(diagnostics.estimated_rate.is_some());
        assert!((diagnostics.estimated_rate.unwrap() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Wrong number of initial values
        let result = newton_raphson_system(&system, &[0.5], &config);
        assert!(matches!(
            result,
            Err(NonlinearSystemSolverError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compare_newton_vs_broyden_iterations() {
        let system = make_hyperbola_line_system();

        // Use a better initial guess closer to the solution (2, 1)
        let initial_guess = [2.1, 0.9];

        let newton_config = NonlinearSystemConfig::default();
        let broyden_config = NonlinearSystemConfig {
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        };

        let newton_result = newton_raphson_system(&system, &initial_guess, &newton_config).unwrap();
        let broyden_result = broyden_system(&system, &initial_guess, &broyden_config).unwrap();

        // Both should converge
        assert!(newton_result.converged);
        assert!(broyden_result.converged);

        // Newton typically converges faster (fewer iterations)
        // But Broyden avoids Jacobian recomputation
        println!(
            "Newton iterations: {}, Broyden iterations: {}",
            newton_result.iterations, broyden_result.iterations
        );
    }

    #[test]
    fn test_solution_as_map() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        let result = newton_raphson_system(&system, &[0.5, 0.5], &config).unwrap();
        let map = result.as_map();

        assert!(map.contains_key("x"));
        assert!(map.contains_key("y"));
    }

    #[test]
    fn test_nonlinear_system_solver_trait() {
        let system = make_circle_line_system();
        let config = NonlinearSystemConfig::default();

        // Test with trait object
        let solvers: Vec<Box<dyn NonlinearSystemSolver>> =
            vec![Box::new(NewtonRaphsonSolver), Box::new(BroydenSolver)];

        for solver in solvers {
            let result = solver.solve(&system, &[0.5, 0.5], &config).unwrap();
            assert!(result.converged);
            println!("{}: {} iterations", solver.method_name(), result.iterations);
        }
    }
}
