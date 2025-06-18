declare namespace wasm_bindgen {
	/* tslint:disable */
	/* eslint-disable */
	export class PhysicsParams {
	  private constructor();
	  free(): void;
	  cart_m: number;
	  m1: number;
	  m2: number;
	  l1_m: number;
	  l2_m: number;
	  g: number;
	}
	export class PhysicsState {
	  private constructor();
	  free(): void;
	  a1: number;
	  a2: number;
	  a1_v: number;
	  a2_v: number;
	  cart_x_m: number;
	  cart_x_v_m: number;
	}
	export class WasmPendulumPhysics {
	  free(): void;
	  constructor(cart_m: number, m1: number, m2: number, l1_m: number, l2_m: number, g: number);
	  reset(): void;
	  get_state_js(): any;
	  get_params_js(): any;
	  update_physics_step(dt: number, force_override: number, simulation_mode_is_observing: boolean): boolean;
	}
	
}

declare type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

declare interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_physicsparams_free: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_cart_m: (a: number) => number;
  readonly __wbg_set_physicsparams_cart_m: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_m1: (a: number) => number;
  readonly __wbg_set_physicsparams_m1: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_m2: (a: number) => number;
  readonly __wbg_set_physicsparams_m2: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_l1_m: (a: number) => number;
  readonly __wbg_set_physicsparams_l1_m: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_l2_m: (a: number) => number;
  readonly __wbg_set_physicsparams_l2_m: (a: number, b: number) => void;
  readonly __wbg_get_physicsparams_g: (a: number) => number;
  readonly __wbg_set_physicsparams_g: (a: number, b: number) => void;
  readonly __wbg_physicsstate_free: (a: number, b: number) => void;
  readonly __wbg_wasmpendulumphysics_free: (a: number, b: number) => void;
  readonly wasmpendulumphysics_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly wasmpendulumphysics_reset: (a: number) => void;
  readonly wasmpendulumphysics_get_state_js: (a: number) => any;
  readonly wasmpendulumphysics_get_params_js: (a: number) => any;
  readonly wasmpendulumphysics_update_physics_step: (a: number, b: number, c: number, d: number) => number;
  readonly __wbg_set_physicsstate_a1: (a: number, b: number) => void;
  readonly __wbg_set_physicsstate_a2: (a: number, b: number) => void;
  readonly __wbg_set_physicsstate_a1_v: (a: number, b: number) => void;
  readonly __wbg_set_physicsstate_a2_v: (a: number, b: number) => void;
  readonly __wbg_set_physicsstate_cart_x_m: (a: number, b: number) => void;
  readonly __wbg_set_physicsstate_cart_x_v_m: (a: number, b: number) => void;
  readonly __wbg_get_physicsstate_a1: (a: number) => number;
  readonly __wbg_get_physicsstate_a2: (a: number) => number;
  readonly __wbg_get_physicsstate_a1_v: (a: number) => number;
  readonly __wbg_get_physicsstate_a2_v: (a: number) => number;
  readonly __wbg_get_physicsstate_cart_x_m: (a: number) => number;
  readonly __wbg_get_physicsstate_cart_x_v_m: (a: number) => number;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
declare function wasm_bindgen (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
