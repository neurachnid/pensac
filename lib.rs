use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::f64::consts::PI; // Import Rust's PI constant for f64

// For debugging: panic messages will go to console.error
#[cfg(feature = "console_error_panic_hook")]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

// Helper to check for NaN/Infinity
fn is_valid_num(val: f64, name: &str) -> bool {
    if !val.is_finite() {
        warn(&format!("Invalid number for {}: {}", name, val));
        return false;
    }
    true
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PhysicsParams {
    pub cart_m: f64,
    pub m1: f64,
    pub m2: f64,
    pub l1_m: f64,
    pub l2_m: f64,
    pub g: f64,
    pub cart_b: f64,
    pub p1_b: f64,
    pub p2_b: f64,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct PhysicsState {
    pub a1: f64,
    pub a2: f64,
    pub a1_v: f64,
    pub a2_v: f64,
    pub cart_x_m: f64,
    pub cart_x_v_m: f64,
}

#[wasm_bindgen]
pub struct WasmPendulumPhysics {
    params: PhysicsParams,
    state: PhysicsState,
}

#[wasm_bindgen]
impl WasmPendulumPhysics {
    #[wasm_bindgen(constructor)]
    pub fn new(cart_m: f64, m1: f64, m2: f64, l1_m: f64, l2_m: f64, g: f64) -> Self {
        let params = PhysicsParams {
            cart_m,
            m1,
            m2,
            l1_m,
            l2_m,
            g,
            cart_b: 0.05,
            p1_b: 0.001,
            p2_b: 0.001,
        };
        let initial_state = Self::get_initial_down_state();
        WasmPendulumPhysics {
            params,
            state: initial_state,
        }
    }

    fn get_initial_down_state() -> PhysicsState {
        PhysicsState {
            a1: 0.0,
            a2: 0.0,
            a1_v: 0.0,
            a2_v: 0.0,
            cart_x_m: 0.0,
            cart_x_v_m: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.state = Self::get_initial_down_state();
    }

    // Returns the current state as a JS object
    pub fn get_state_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.state).unwrap_or(JsValue::NULL)
    }
    
    // Returns the current params as a JS object
    pub fn get_params_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.params).unwrap_or(JsValue::NULL)
    }

    // Core physics update, returns true if successful
    pub fn update_physics_step(&mut self, dt: f64, force_override: f64, simulation_mode_is_observing: bool) -> bool {
        if !is_valid_num(force_override, "force_override") { return false; }

        let fx = force_override;

        let k1 = match compute_derivatives(self.params, self.state, fx) {
            Some(k) => k,
            None => return false,
        };
        let temp_state_k2 = state_add_scaled(&self.state, &k1, dt / 2.0);
        let k2 = match compute_derivatives(self.params, temp_state_k2, fx) {
            Some(k) => k,
            None => return false,
        };
        let temp_state_k3 = state_add_scaled(&self.state, &k2, dt / 2.0);
        let k3 = match compute_derivatives(self.params, temp_state_k3, fx) {
            Some(k) => k,
            None => return false,
        };
        let temp_state_k4 = state_add_scaled(&self.state, &k3, dt);
        let k4 = match compute_derivatives(self.params, temp_state_k4, fx) {
            Some(k) => k,
            None => return false,
        };

        let dt_over_6 = dt / 6.0;
        let new_cart_x_m = self.state.cart_x_m + dt_over_6 * (k1.cart_x_m + 2.0 * k2.cart_x_m + 2.0 * k3.cart_x_m + k4.cart_x_m);
        let new_cart_x_v_m = self.state.cart_x_v_m + dt_over_6 * (k1.cart_x_v_m + 2.0 * k2.cart_x_v_m + 2.0 * k3.cart_x_v_m + k4.cart_x_v_m);
        let new_a1 = self.state.a1 + dt_over_6 * (k1.a1 + 2.0 * k2.a1 + 2.0 * k3.a1 + k4.a1);
        let new_a2 = self.state.a2 + dt_over_6 * (k1.a2 + 2.0 * k2.a2 + 2.0 * k3.a2 + k4.a2);
        let new_a1_v = self.state.a1_v + dt_over_6 * (k1.a1_v + 2.0 * k2.a1_v + 2.0 * k3.a1_v + k4.a1_v);
        let new_a2_v = self.state.a2_v + dt_over_6 * (k1.a2_v + 2.0 * k2.a2_v + 2.0 * k3.a2_v + k4.a2_v);

        for (val, name) in [
            (new_cart_x_v_m, "new_cart_x_v_m"),
            (new_a1_v, "new_a1_v"),
            (new_a2_v, "new_a2_v"),
            (new_cart_x_m, "new_cart_x_m"),
            (new_a1, "new_a1"),
            (new_a2, "new_a2"),
        ] {
            if !is_valid_num(val, name) { return false; }
        }

        const MAX_CART_VELOCITY: f64 = 15.0;
        const MAX_ANGULAR_VELOCITY: f64 = 25.0;

        self.state.cart_x_v_m = new_cart_x_v_m.max(-MAX_CART_VELOCITY).min(MAX_CART_VELOCITY);
        self.state.a1_v = new_a1_v.max(-MAX_ANGULAR_VELOCITY).min(MAX_ANGULAR_VELOCITY);
        self.state.a2_v = new_a2_v.max(-MAX_ANGULAR_VELOCITY).min(MAX_ANGULAR_VELOCITY);

        if simulation_mode_is_observing {
            self.state.cart_x_m = new_cart_x_m;
        } else {
            self.state.cart_x_m = new_cart_x_m.max(-100.0).min(100.0);
        }
        self.state.a1 = new_a1;
        self.state.a2 = new_a2;

        true
    }
}

// Helper: 3x3 Matrix Inversion (Gaussian Elimination)
fn mat_inv_3x3(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut m = a;
    let mut inv = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    for i in 0..3 {
        let mut pivot = i;
        for j in (i + 1)..3 {
            if m[j][i].abs() > m[pivot][i].abs() {
                pivot = j;
            }
        }

        if pivot != i {
            m.swap(i, pivot);
            inv.swap(i, pivot);
        }

        let div = m[i][i];
        if div.abs() < 1e-12 { return None; } // Singular matrix

        for j in i..3 { m[i][j] /= div; }
        for j in 0..3 { inv[i][j] /= div; }

        for j in 0..3 {
            if i != j {
                let mult = m[j][i];
                for k in i..3 { m[j][k] -= mult * m[i][k]; }
                for k in 0..3 { inv[j][k] -= mult * inv[i][k]; }
            }
        }
    }
    Some(inv)
}

// Helper: 3x3 Matrix-Vector Multiplication
fn multiply_matrix_vector_3x3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// Compute accelerations q_ddot = [x_ddot, a1_ddot, a2_ddot]
fn compute_q_ddot(params: PhysicsParams, state: PhysicsState, fx: f64) -> Option<[f64; 3]> {
    let PhysicsParams { cart_m, m1, m2, l1_m, l2_m, g, cart_b, p1_b, p2_b } = params;
    let PhysicsState { a1, a2, a1_v, a2_v, cart_x_m: _, cart_x_v_m } = state;

    let state_vars = [a1, a2, a1_v, a2_v, cart_x_v_m];
    let var_names = ["a1", "a2", "a1_v", "a2_v", "cart_x_v_m"];
    for (i, &val) in state_vars.iter().enumerate() {
        if !is_valid_num(val, var_names[i]) { return None; }
    }

    let clamped_a1 = a1.max(-4.0 * PI).min(4.0 * PI);
    let clamped_a2 = a2.max(-4.0 * PI).min(4.0 * PI);
    let clamped_a1_v = a1_v.max(-50.0).min(50.0);
    let clamped_a2_v = a2_v.max(-50.0).min(50.0);
    let _clamped_cart_x_v_m = cart_x_v_m.max(-20.0).min(20.0);

    let c1 = clamped_a1.cos(); let s1 = clamped_a1.sin();
    let c2 = clamped_a2.cos(); let s2 = clamped_a2.sin();
    let c12 = (clamped_a1 - clamped_a2).cos();
    let s12 = (clamped_a1 - clamped_a2).sin();

    let trig_values = [c1, s1, c2, s2, c12, s12];
    let trig_names = ["c1", "s1", "c2", "s2", "c12", "s12"];
    for (i, &val) in trig_values.iter().enumerate() {
        if !is_valid_num(val, trig_names[i]) { return None; }
    }

    let m_total_pend = m1 + m2;

    let m_matrix = [
        [cart_m + m_total_pend, m_total_pend * l1_m * c1, m2 * l2_m * c2],
        [m_total_pend * l1_m * c1, m_total_pend * l1_m * l1_m, m2 * l1_m * l2_m * c12],
        [m2 * l2_m * c2, m2 * l1_m * l2_m * c12, m2 * l2_m * l2_m],
    ];

    let b_vector = [
        fx
            - cart_b * cart_x_v_m
            + m_total_pend * l1_m * s1 * clamped_a1_v * clamped_a1_v
            + m2 * l2_m * s2 * clamped_a2_v * clamped_a2_v,
        -p1_b * clamped_a1_v
            - m2 * l1_m * l2_m * s12 * clamped_a2_v * clamped_a2_v
            - m_total_pend * g * l1_m * s1,
        -p2_b * clamped_a2_v
            + m2 * l1_m * l2_m * s12 * clamped_a1_v * clamped_a1_v
            - m2 * g * l2_m * s2,
    ];

    for r in 0..3 {
        for c in 0..3 { if !is_valid_num(m_matrix[r][c], &format!("M[{}][{}]", r, c)) { return None; } }
        if !is_valid_num(b_vector[r], &format!("b[{}]", r)) { return None; }
    }

    if let Some(m_inv) = mat_inv_3x3(m_matrix) {
        let q_ddot = multiply_matrix_vector_3x3(m_inv, b_vector);
        for (i, &val) in q_ddot.iter().enumerate() {
            if !is_valid_num(val, &format!("q_ddot[{}]", i)) { return None; }
        }
        let max_accel = 1000.0;
        let q_ddot_clamped = [
            q_ddot[0].max(-max_accel).min(max_accel),
            q_ddot[1].max(-max_accel).min(max_accel),
            q_ddot[2].max(-max_accel).min(max_accel),
        ];
        Some(q_ddot_clamped)
    } else {
        warn("Matrix inversion failed in Wasm updatePhysics");
        None
    }
}

// Compute derivatives of the state for RK4
fn compute_derivatives(params: PhysicsParams, state: PhysicsState, fx: f64) -> Option<PhysicsState> {
    if let Some(q_ddot) = compute_q_ddot(params, state, fx) {
        Some(PhysicsState {
            a1: state.a1_v,
            a2: state.a2_v,
            a1_v: q_ddot[1],
            a2_v: q_ddot[2],
            cart_x_m: state.cart_x_v_m,
            cart_x_v_m: q_ddot[0],
        })
    } else {
        None
    }
}

fn state_add_scaled(base: &PhysicsState, delta: &PhysicsState, scale: f64) -> PhysicsState {
    PhysicsState {
        a1: base.a1 + delta.a1 * scale,
        a2: base.a2 + delta.a2 * scale,
        a1_v: base.a1_v + delta.a1_v * scale,
        a2_v: base.a2_v + delta.a2_v * scale,
        cart_x_m: base.cart_x_m + delta.cart_x_m * scale,
        cart_x_v_m: base.cart_x_v_m + delta.cart_x_v_m * scale,
    }
}
