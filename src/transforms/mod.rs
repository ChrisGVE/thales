//! Coordinate system transformations.
//!
//! Provides transformations between different coordinate systems
//! (Cartesian, Polar, Spherical, Cylindrical) and complex number operations.
//! Also exports [`CoordSystem`] and the [`jacobian`] sub-module for symbolic
//! Jacobian matrix and volume-element construction.

mod cartesian;
mod complex;
pub mod jacobian;
mod polar;
mod transform2d;

pub use cartesian::{Cartesian2D, Cartesian3D};
pub use complex::{decompose_complex_equation, separate_real_imag, ComplexOps};
pub use polar::{Cylindrical, Polar, Spherical};
pub use transform2d::{Rotation3D, Transform2D};

/// Built-in coordinate systems recognised by [`jacobian::volume_element`].
///
/// The `Custom` variant is a pass-through for caller-supplied parametric maps;
/// [`jacobian::volume_element`] returns `1` for it (the caller is expected to
/// use [`jacobian::jacobian_determinant`] directly on their own forward map).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordSystem {
    /// 2-D Cartesian (x, y).  Volume element: 1.
    Cartesian2D,
    /// 2-D polar (r, θ).  Volume element: r.
    Polar2D,
    /// 3-D Cartesian (x, y, z).  Volume element: 1.
    Cartesian3D,
    /// 3-D cylindrical (ρ, φ, z).  Volume element: ρ.
    Cylindrical,
    /// 3-D spherical (ρ, θ, φ).  Volume element: ρ²·sin(φ).
    Spherical,
    /// 2-D parabolic (u, v).  Volume element: u² + v².
    Parabolic2D,
    /// 2-D elliptic (μ, ν) with focal half-distance a.
    /// Volume element: a·√(sinh²μ + sin²ν).
    Elliptic2D,
    /// Caller-defined curvilinear system.  Volume element placeholder: 1.
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // --- identity ---

    #[test]
    fn identity_leaves_point_unchanged() {
        let p = Cartesian2D::new(3.0, 7.0);
        let result = Transform2D::identity().apply(p);
        assert!(approx_eq(result.x, 3.0));
        assert!(approx_eq(result.y, 7.0));
    }

    // --- translation ---

    #[test]
    fn translation_moves_origin_to_offset() {
        let t = Transform2D::translation(3.0, 4.0);
        let result = t.apply(Cartesian2D::new(0.0, 0.0));
        assert!(approx_eq(result.x, 3.0));
        assert!(approx_eq(result.y, 4.0));
    }

    #[test]
    fn translation_adds_offset_to_point() {
        let t = Transform2D::translation(1.0, -2.0);
        let result = t.apply(Cartesian2D::new(5.0, 6.0));
        assert!(approx_eq(result.x, 6.0));
        assert!(approx_eq(result.y, 4.0));
    }

    // --- rotation ---

    #[test]
    fn rotation_90_maps_unit_x_to_unit_y() {
        let t = Transform2D::rotation(FRAC_PI_2);
        let result = t.apply(Cartesian2D::new(1.0, 0.0));
        assert!(approx_eq(result.x, 0.0));
        assert!(approx_eq(result.y, 1.0));
    }

    #[test]
    fn rotation_zero_is_identity() {
        let t = Transform2D::rotation(0.0);
        let p = Cartesian2D::new(3.0, 4.0);
        let result = t.apply(p);
        assert!(approx_eq(result.x, 3.0));
        assert!(approx_eq(result.y, 4.0));
    }

    // --- scaling ---

    #[test]
    fn scaling_multiplies_coordinates() {
        let t = Transform2D::scaling(2.0, 2.0);
        let result = t.apply(Cartesian2D::new(2.0, 3.0));
        assert!(approx_eq(result.x, 4.0));
        assert!(approx_eq(result.y, 6.0));
    }

    #[test]
    fn scaling_non_uniform() {
        let t = Transform2D::scaling(3.0, 0.5);
        let result = t.apply(Cartesian2D::new(2.0, 4.0));
        assert!(approx_eq(result.x, 6.0));
        assert!(approx_eq(result.y, 2.0));
    }

    // --- compose ---

    #[test]
    fn compose_identity_is_neutral() {
        let t = Transform2D::translation(5.0, -3.0);
        let composed = t.compose(&Transform2D::identity());
        let p = Cartesian2D::new(1.0, 2.0);
        let direct = t.apply(p);
        let via_compose = composed.apply(p);
        assert!(approx_eq(direct.x, via_compose.x));
        assert!(approx_eq(direct.y, via_compose.y));
    }

    #[test]
    fn compose_translate_then_rotate_differs_from_rotate_then_translate() {
        let tr = Transform2D::translation(1.0, 0.0);
        let rot = Transform2D::rotation(FRAC_PI_2);

        // rotate first, then translate: rot.compose(tr) applies tr then rot
        let rot_then_tr = rot.compose(&tr);
        // translate first, then rotate: tr.compose(rot) applies rot then tr
        let tr_then_rot = tr.compose(&rot);

        let p = Cartesian2D::new(0.0, 0.0);
        let r1 = rot_then_tr.apply(p);
        let r2 = tr_then_rot.apply(p);

        // Results must differ, confirming non-commutativity
        assert!(!approx_eq(r1.x, r2.x) || !approx_eq(r1.y, r2.y));
    }

    #[test]
    fn compose_two_translations_adds_offsets() {
        let t1 = Transform2D::translation(1.0, 2.0);
        let t2 = Transform2D::translation(3.0, 4.0);
        let composed = t1.compose(&t2);
        let result = composed.apply(Cartesian2D::new(0.0, 0.0));
        assert!(approx_eq(result.x, 4.0));
        assert!(approx_eq(result.y, 6.0));
    }

    // --- nth_root ---

    const COMPLEX_EPS: f64 = 1e-10;

    fn complex_approx_eq(a: num_complex::Complex64, b: num_complex::Complex64) -> bool {
        (a.re - b.re).abs() < COMPLEX_EPS && (a.im - b.im).abs() < COMPLEX_EPS
    }

    #[test]
    fn nth_root_square_roots_of_one_are_one_and_minus_one() {
        let z = num_complex::Complex64::new(1.0, 0.0);
        let roots = ComplexOps::nth_root(z, 2);
        assert_eq!(roots.len(), 2);
        // roots must be 1 and -1 (in some order)
        let has_pos_one = roots
            .iter()
            .any(|r| complex_approx_eq(*r, num_complex::Complex64::new(1.0, 0.0)));
        let has_neg_one = roots
            .iter()
            .any(|r| complex_approx_eq(*r, num_complex::Complex64::new(-1.0, 0.0)));
        assert!(has_pos_one, "missing root +1");
        assert!(has_neg_one, "missing root -1");
    }

    #[test]
    fn nth_root_cube_roots_of_one_lie_on_unit_circle_spaced_120_degrees_apart() {
        use std::f64::consts::PI;
        let z = num_complex::Complex64::new(1.0, 0.0);
        let roots = ComplexOps::nth_root(z, 3);
        assert_eq!(roots.len(), 3);
        // All roots must have magnitude 1
        for r in &roots {
            assert!(
                (r.norm() - 1.0).abs() < COMPLEX_EPS,
                "root magnitude != 1: {r}"
            );
        }
        // Angles must be 0, 2π/3, 4π/3
        let expected_angles = [0.0_f64, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
        for expected in expected_angles {
            let found = roots.iter().any(|r| {
                let angle = r.arg().rem_euclid(2.0 * PI);
                (angle - expected).abs() < COMPLEX_EPS
            });
            assert!(found, "missing root at angle {expected}");
        }
    }

    #[test]
    fn nth_root_sqrt_of_minus_one_gives_i_and_minus_i() {
        let z = num_complex::Complex64::new(-1.0, 0.0);
        let roots = ComplexOps::nth_root(z, 2);
        assert_eq!(roots.len(), 2);
        let has_i = roots
            .iter()
            .any(|r| complex_approx_eq(*r, num_complex::Complex64::new(0.0, 1.0)));
        let has_neg_i = roots
            .iter()
            .any(|r| complex_approx_eq(*r, num_complex::Complex64::new(0.0, -1.0)));
        assert!(has_i, "missing root +i");
        assert!(has_neg_i, "missing root -i");
    }

    #[test]
    fn nth_root_fourth_roots_of_16_include_two_and_minus_two_and_two_i() {
        let z = num_complex::Complex64::new(16.0, 0.0);
        let roots = ComplexOps::nth_root(z, 4);
        assert_eq!(roots.len(), 4);
        let expected: &[num_complex::Complex64] = &[
            num_complex::Complex64::new(2.0, 0.0),
            num_complex::Complex64::new(-2.0, 0.0),
            num_complex::Complex64::new(0.0, 2.0),
            num_complex::Complex64::new(0.0, -2.0),
        ];
        for exp in expected {
            let found = roots.iter().any(|r| complex_approx_eq(*r, *exp));
            assert!(found, "missing expected root {exp}");
        }
    }

    #[test]
    fn nth_root_all_roots_raised_to_n_recover_original() {
        let cases: &[(num_complex::Complex64, i32)] = &[
            (num_complex::Complex64::new(1.0, 0.0), 2),
            (num_complex::Complex64::new(1.0, 0.0), 3),
            (num_complex::Complex64::new(-1.0, 0.0), 2),
            (num_complex::Complex64::new(16.0, 0.0), 4),
            (num_complex::Complex64::new(3.0, 4.0), 3),
        ];
        for &(z, n) in cases {
            let roots = ComplexOps::nth_root(z, n);
            assert_eq!(roots.len(), n as usize);
            for root in roots {
                let recovered = root.powi(n);
                assert!(
                    complex_approx_eq(recovered, z),
                    "root^{n} = {recovered} != {z} (root = {root})"
                );
            }
        }
    }

    #[test]
    fn nth_root_non_positive_n_returns_empty() {
        let z = num_complex::Complex64::new(1.0, 0.0);
        assert!(ComplexOps::nth_root(z, 0).is_empty());
        assert!(ComplexOps::nth_root(z, -1).is_empty());
    }
}
