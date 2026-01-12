//! # Coordinate System Transformations Guide
//!
//! This guide covers coordinate system transformations in thales, from 2D Cartesian-Polar
//! conversions to 3D transformations between Cartesian, Spherical, and Cylindrical systems.
//! Learn how to convert between coordinate systems, work with complex numbers in polar form,
//! and apply these transformations to real-world problems.
//!
//! ## Quick Start: 2D Transformations
//!
//! The simplest use case is converting between Cartesian and Polar coordinates:
//!
//! ```rust,ignore
//! use thales::{Cartesian2D, Polar};
//!
//! // Create a point in Cartesian coordinates
//! let point = Cartesian2D::new(3.0, 4.0);
//!
//! // Convert to Polar coordinates
//! let polar = point.to_polar();
//! assert!((polar.r - 5.0).abs() < 1e-10);           // r = √(3² + 4²) = 5
//! assert!((polar.theta - 0.927295218).abs() < 1e-6); // θ = arctan(4/3) ≈ 0.927 rad
//!
//! // Convert back to Cartesian (round-trip)
//! let back = polar.to_cartesian();
//! assert!((back.x - 3.0).abs() < 1e-10);
//! assert!((back.y - 4.0).abs() < 1e-10);
//! ```
//!
//! All transformations are bidirectional and preserve geometric properties through
//! round-trip conversions.
//!
//! ## 2D Coordinate Systems
//!
//! ### Cartesian Coordinates
//!
//! Cartesian coordinates represent points using perpendicular x and y axes:
//!
//! ```rust,ignore
//! use thales::Cartesian2D;
//!
//! let origin = Cartesian2D::new(0.0, 0.0);
//! let point = Cartesian2D::new(3.0, 4.0);
//!
//! // Direct field access
//! println!("x = {}, y = {}", point.x, point.y);
//! ```
//!
//! Cartesian coordinates are ideal for:
//! - Linear motion and forces
//! - Rectangular geometries
//! - Grid-based computations
//!
//! ### Polar Coordinates
//!
//! Polar coordinates represent points using radius r and angle θ:
//!
//! ```rust,ignore
//! use thales::{Polar, Cartesian2D};
//! use std::f64::consts::PI;
//!
//! // Create point at radius 5, angle π/4 (45 degrees)
//! let polar = Polar::new(5.0, PI / 4.0);
//!
//! // Convert to Cartesian
//! let cartesian = polar.to_cartesian();
//! assert!((cartesian.x - 3.535533906).abs() < 1e-6); // 5 * cos(π/4)
//! assert!((cartesian.y - 3.535533906).abs() < 1e-6); // 5 * sin(π/4)
//! ```
//!
//! Polar coordinates are ideal for:
//! - Circular motion and rotation
//! - Angular velocities
//! - Radar and navigation systems
//!
//! ### When to Use Each System
//!
//! | Scenario | Recommended System | Reason |
//! |----------|-------------------|---------|
//! | Robot motion on flat surface | Cartesian | Linear coordinates match actuators |
//! | Satellite orbit calculations | Polar | Natural for circular paths |
//! | Screen pixel positioning | Cartesian | Matches display hardware |
//! | Radar tracking | Polar | Direct r and θ measurements |
//!
//! ## 3D Coordinate Systems
//!
//! ### Cartesian 3D Coordinates
//!
//! Three-dimensional Cartesian coordinates extend 2D with a z-axis:
//!
//! ```rust,ignore
//! use thales::Cartesian3D;
//!
//! let point = Cartesian3D::new(1.0, 1.0, 1.0);
//! println!("Point at ({}, {}, {})", point.x, point.y, point.z);
//! ```
//!
//! ### Spherical Coordinates
//!
//! Spherical coordinates use radius r, polar angle θ, and azimuthal angle φ:
//!
//! ```rust,ignore
//! use thales::{Cartesian3D, Spherical};
//!
//! // Convert Cartesian to Spherical
//! let cart = Cartesian3D::new(1.0, 1.0, 1.0);
//! let spherical = cart.to_spherical();
//!
//! // r = √(x² + y² + z²)
//! assert!((spherical.r - 1.732050808).abs() < 1e-6);
//!
//! // Round-trip conversion
//! let back = spherical.to_cartesian();
//! assert!((back.x - 1.0).abs() < 1e-10);
//! assert!((back.y - 1.0).abs() < 1e-10);
//! assert!((back.z - 1.0).abs() < 1e-10);
//! ```
//!
//! Spherical coordinates are ideal for:
//! - Astronomy and celestial mechanics
//! - Electromagnetic fields
//! - Global positioning (latitude/longitude)
//!
//! ### Cylindrical Coordinates
//!
//! Cylindrical coordinates combine polar coordinates (r, θ) with height z:
//!
//! ```rust,ignore
//! use thales::{Cartesian3D, Cylindrical};
//!
//! // Convert Cartesian to Cylindrical
//! let cart = Cartesian3D::new(3.0, 4.0, 5.0);
//! let cylindrical = cart.to_cylindrical();
//!
//! // rho = √(x² + y²), z remains the same
//! assert!((cylindrical.rho - 5.0).abs() < 1e-10);
//! assert!((cylindrical.z - 5.0).abs() < 1e-10);
//!
//! // Convert back to Cartesian
//! let back = cylindrical.to_cartesian();
//! assert!((back.x - 3.0).abs() < 1e-10);
//! assert!((back.y - 4.0).abs() < 1e-10);
//! assert!((back.z - 5.0).abs() < 1e-10);
//! ```
//!
//! Cylindrical coordinates are ideal for:
//! - Objects with rotational symmetry
//! - Pipes and cylinders
//! - Rotating machinery
//!
//! ### 3D System Selection
//!
//! | Scenario | Recommended System | Reason |
//! |----------|-------------------|---------|
//! | Satellite orbits | Spherical | Natural for radial distance |
//! | Drilling operations | Cylindrical | Rotational + depth |
//! | 3D printing | Cartesian | Matches linear actuators |
//! | Antenna radiation patterns | Spherical | Angular distribution |
//!
//! ## Complex Number Operations
//!
//! Complex numbers can be represented in rectangular (a + bi) or polar (r∠θ) form.
//! The [`ComplexOps`](crate::ComplexOps) trait provides polar form operations:
//!
//! ### Polar Form Conversions
//!
//! ```rust,ignore
//! use thales::{ComplexOps, Polar};
//! use num_complex::Complex64;
//!
//! let z = Complex64::new(3.0, 4.0); // 3 + 4i
//!
//! // Convert to polar form
//! let polar = ComplexOps::to_polar(z);
//! assert!((polar.r - 5.0).abs() < 1e-10);           // |z| = √(3² + 4²) = 5
//! assert!((polar.theta - 0.927295218).abs() < 1e-6); // arg(z) = arctan(4/3)
//!
//! // Convert back to rectangular form
//! let back = ComplexOps::from_polar(polar);
//! assert!((back.re - 3.0).abs() < 1e-10);
//! assert!((back.im - 4.0).abs() < 1e-10);
//! ```
//!
//! ### De Moivre's Theorem
//!
//! De Moivre's theorem states that (r∠θ)ⁿ = rⁿ∠(nθ):
//!
//! ```rust,ignore
//! use thales::ComplexOps;
//! use num_complex::Complex64;
//!
//! let z = Complex64::new(1.0, 1.0); // 1 + i
//!
//! // Compute z² using De Moivre's theorem
//! let z_squared = ComplexOps::de_moivre(z, 2.0);
//!
//! // (1 + i)² = 1 + 2i - 1 = 2i
//! assert!((z_squared.re - 0.0).abs() < 1e-10);
//! assert!((z_squared.im - 2.0).abs() < 1e-10);
//!
//! // Compute z^(1/2) (square root)
//! let z_sqrt = ComplexOps::de_moivre(z, 0.5);
//! assert!((z_sqrt.re - 1.098684).abs() < 1e-5);
//! assert!((z_sqrt.im - 0.455090).abs() < 1e-5);
//! ```
//!
//! De Moivre's theorem is particularly useful for:
//! - Computing integer powers of complex numbers
//! - Finding nth roots of complex numbers
//! - Trigonometric identities
//!
//! ### Complex Conjugate and Modulus
//!
//! The `num_complex` crate provides standard complex operations:
//!
//! ```rust,ignore
//! use num_complex::Complex64;
//!
//! let z = Complex64::new(3.0, 4.0); // 3 + 4i
//!
//! // Complex conjugate: z* = 3 - 4i
//! let conj = z.conj();
//! assert_eq!(conj.re, 3.0);
//! assert_eq!(conj.im, -4.0);
//!
//! // Modulus (magnitude): |z| = √(3² + 4²) = 5
//! let modulus = z.norm();
//! assert!((modulus - 5.0).abs() < 1e-10);
//!
//! // Complex multiplication
//! let w = Complex64::new(1.0, 2.0);
//! let product = z * w; // (3+4i)(1+2i) = 3+6i+4i-8 = -5+10i
//! assert_eq!(product.re, -5.0);
//! assert_eq!(product.im, 10.0);
//! ```
//!
//! ## Round-Trip Conversions
//!
//! All coordinate transformations preserve geometric information through round-trip conversions:
//!
//! ### 2D Round-Trip
//!
//! ```rust,ignore
//! use thales::{Cartesian2D, Polar};
//!
//! let original = Cartesian2D::new(3.0, 4.0);
//! let polar = original.to_polar();
//! let back = polar.to_cartesian();
//!
//! assert!((back.x - original.x).abs() < 1e-10);
//! assert!((back.y - original.y).abs() < 1e-10);
//! ```
//!
//! ### 3D Spherical Round-Trip
//!
//! ```rust,ignore
//! use thales::{Cartesian3D, Spherical};
//!
//! let original = Cartesian3D::new(1.0, 2.0, 3.0);
//! let spherical = original.to_spherical();
//! let back = spherical.to_cartesian();
//!
//! assert!((back.x - original.x).abs() < 1e-10);
//! assert!((back.y - original.y).abs() < 1e-10);
//! assert!((back.z - original.z).abs() < 1e-10);
//! ```
//!
//! ### 3D Cylindrical Round-Trip
//!
//! ```rust,ignore
//! use thales::{Cartesian3D, Cylindrical};
//!
//! let original = Cartesian3D::new(3.0, 4.0, 5.0);
//! let cylindrical = original.to_cylindrical();
//! let back = cylindrical.to_cartesian();
//!
//! assert!((back.x - original.x).abs() < 1e-10);
//! assert!((back.y - original.y).abs() < 1e-10);
//! assert!((back.z - original.z).abs() < 1e-10);
//! ```
//!
//! Numerical precision is maintained within floating-point error bounds (typically < 1e-10).
//!
//! ## Common Use Cases
//!
//! ### Use Case 1: Robot Navigation
//!
//! Convert sensor data (polar) to control coordinates (Cartesian):
//!
//! ```rust,ignore
//! use thales::{Polar, Cartesian2D};
//!
//! // Sensor detects obstacle at 5m, 45 degrees
//! let obstacle_polar = Polar::new(5.0, 0.785398); // π/4 radians
//! let obstacle_cart = obstacle_polar.to_cartesian();
//!
//! // Plan path in Cartesian coordinates
//! println!("Obstacle at ({:.2}, {:.2})", obstacle_cart.x, obstacle_cart.y);
//! ```
//!
//! ### Use Case 2: Physics Simulation
//!
//! Calculate orbital mechanics using spherical coordinates:
//!
//! ```rust,ignore
//! use thales::{Cartesian3D, Spherical};
//!
//! // Satellite position in Cartesian
//! let position = Cartesian3D::new(6378.0, 0.0, 0.0); // km from Earth center
//!
//! // Convert to spherical for orbital calculations
//! let spherical = position.to_spherical();
//! let altitude = spherical.r - 6371.0; // Subtract Earth radius
//!
//! println!("Satellite altitude: {:.1} km", altitude);
//! ```
//!
//! ### Use Case 3: Signal Processing
//!
//! Analyze frequency response using complex polar form:
//!
//! ```rust,ignore
//! use thales::ComplexOps;
//! use num_complex::Complex64;
//!
//! // Frequency response: H(ω) = 1 / (1 + iω)
//! let omega = 1.0;
//! let h = Complex64::new(1.0, 0.0) / Complex64::new(1.0, omega);
//!
//! // Magnitude and phase in polar form
//! let polar = ComplexOps::to_polar(h);
//! println!("Magnitude: {:.3}, Phase: {:.3} rad", polar.r, polar.theta);
//! ```
//!
//! ### Use Case 4: 3D Graphics
//!
//! Convert camera position from spherical to Cartesian:
//!
//! ```rust,ignore
//! use thales::{Spherical, Cartesian3D};
//! use std::f64::consts::PI;
//!
//! // Camera at radius 10, looking down at 30° from horizontal
//! let camera_spherical = Spherical::new(10.0, PI/6.0, PI/4.0);
//! let camera_position = camera_spherical.to_cartesian();
//!
//! println!("Camera at ({:.2}, {:.2}, {:.2})",
//!          camera_position.x, camera_position.y, camera_position.z);
//! ```
//!
//! ### Use Case 5: Antenna Design
//!
//! Calculate radiation pattern in cylindrical coordinates:
//!
//! ```rust,ignore
//! use thales::{Cylindrical, Cartesian3D};
//! use std::f64::consts::PI;
//!
//! // Antenna element at radius 0.5m, 30° angle, height 2m
//! let element = Cylindrical::new(0.5, PI/6.0, 2.0);
//! let position = element.to_cartesian();
//!
//! println!("Element position: ({:.3}, {:.3}, {:.3})",
//!          position.x, position.y, position.z);
//! ```
//!
//! ## Performance Considerations
//!
//! All coordinate transformations are O(1) operations involving a fixed number
//! of trigonometric functions:
//!
//! - Cartesian → Polar: 1 atan2, 1 hypot (√)
//! - Polar → Cartesian: 1 sin, 1 cos
//! - Cartesian3D → Spherical: 1 atan2, 2 hypot (√)
//! - Spherical → Cartesian3D: 2 sin, 2 cos
//! - Cartesian3D → Cylindrical: 1 atan2, 1 hypot (√)
//! - Cylindrical → Cartesian3D: 1 sin, 1 cos
//!
//! For performance-critical applications:
//! 1. Minimize coordinate system changes within tight loops
//! 2. Batch transformations to improve cache locality
//! 3. Use Cartesian coordinates for linear algebra operations
//! 4. Consider precomputing common angles (sin/cos lookup tables)
//!
//! ## See Also
//!
//! - [`crate::Cartesian2D`] - 2D Cartesian coordinate representation
//! - [`crate::Polar`] - 2D polar coordinate representation
//! - [`crate::Cartesian3D`] - 3D Cartesian coordinate representation
//! - [`crate::Spherical`] - 3D spherical coordinate representation
//! - [`crate::Cylindrical`] - 3D cylindrical coordinate representation
//! - [`crate::ComplexOps`] - Complex number operations in polar form
//! - [`num_complex::Complex64`] - Complex number representation (external crate)
