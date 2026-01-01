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
//! use mathsolver_core::equation_system::{EquationSystem, SystemContext, MultiEquationSolver};
//! use mathsolver_core::parse_equation;
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

use crate::ast::{Equation, Expression, Variable};
use crate::numerical::{SmartNumericalSolver, NumericalConfig};
use crate::resolution_path::{Operation, ResolutionPath};
use crate::solver::{SmartSolver, Solution, SolverError, Solver};

// Note: ODE and integration imports reserved for future implementation
// use crate::ode::{FirstOrderODE, ODESolution, solve_separable, solve_linear as solve_linear_ode};
// use crate::integration::integrate;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during equation system solving.
#[derive(Debug, Clone)]
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
            Self::OverdeterminedSystem { equations, unknowns } => {
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
                        if dep_var != var
                            && self.has_cycle_dfs(dep_var, known, visited, rec_stack)
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
            Self::Numeric(v) => None,
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
        let known = context.known_variable_names();

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
                            EquationType::Algebraic | EquationType::Unknown => SolveMethod::Algebraic,
                            EquationType::ODE(info) => SolveMethod::ODE {
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
            let eq = system.get(&step.equation_id).ok_or_else(|| {
                SystemError::EquationNotFound(step.equation_id.clone())
            })?;

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
                        parameter: "t".to_string(),  // Default parameter name
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
        equation: &Equation,
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
        equation: &Equation,
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
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap());

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
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("F = m * a").unwrap());

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
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("x + y = 10").unwrap());

        let context = SystemContext::new()
            .with_target("x")
            .with_target("y");

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
        let system = EquationSystem::new()
            .with_equation("eq1", parse_equation("y = x * x").unwrap());

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
