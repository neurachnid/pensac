let wasm;

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const PhysicsParamsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_physicsparams_free(ptr >>> 0, 1));

export class PhysicsParams {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PhysicsParamsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_physicsparams_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get cart_m() {
        const ret = wasm.__wbg_get_physicsparams_cart_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set cart_m(arg0) {
        wasm.__wbg_set_physicsparams_cart_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get m1() {
        const ret = wasm.__wbg_get_physicsparams_m1(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set m1(arg0) {
        wasm.__wbg_set_physicsparams_m1(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get m2() {
        const ret = wasm.__wbg_get_physicsparams_m2(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set m2(arg0) {
        wasm.__wbg_set_physicsparams_m2(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get l1_m() {
        const ret = wasm.__wbg_get_physicsparams_l1_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set l1_m(arg0) {
        wasm.__wbg_set_physicsparams_l1_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get l2_m() {
        const ret = wasm.__wbg_get_physicsparams_l2_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set l2_m(arg0) {
        wasm.__wbg_set_physicsparams_l2_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get g() {
        const ret = wasm.__wbg_get_physicsparams_g(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set g(arg0) {
        wasm.__wbg_set_physicsparams_g(this.__wbg_ptr, arg0);
    }
}

const PhysicsStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_physicsstate_free(ptr >>> 0, 1));

export class PhysicsState {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PhysicsStateFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_physicsstate_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get a1() {
        const ret = wasm.__wbg_get_physicsparams_cart_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a1(arg0) {
        wasm.__wbg_set_physicsparams_cart_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get a2() {
        const ret = wasm.__wbg_get_physicsparams_m1(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a2(arg0) {
        wasm.__wbg_set_physicsparams_m1(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get a1_v() {
        const ret = wasm.__wbg_get_physicsparams_m2(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a1_v(arg0) {
        wasm.__wbg_set_physicsparams_m2(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get a2_v() {
        const ret = wasm.__wbg_get_physicsparams_l1_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a2_v(arg0) {
        wasm.__wbg_set_physicsparams_l1_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get cart_x_m() {
        const ret = wasm.__wbg_get_physicsparams_l2_m(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set cart_x_m(arg0) {
        wasm.__wbg_set_physicsparams_l2_m(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get cart_x_v_m() {
        const ret = wasm.__wbg_get_physicsparams_g(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set cart_x_v_m(arg0) {
        wasm.__wbg_set_physicsparams_g(this.__wbg_ptr, arg0);
    }
}

const WasmPendulumPhysicsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmpendulumphysics_free(ptr >>> 0, 1));

export class WasmPendulumPhysics {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmPendulumPhysicsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmpendulumphysics_free(ptr, 0);
    }
    /**
     * @param {number} cart_m
     * @param {number} m1
     * @param {number} m2
     * @param {number} l1_m
     * @param {number} l2_m
     * @param {number} g
     */
    constructor(cart_m, m1, m2, l1_m, l2_m, g) {
        const ret = wasm.wasmpendulumphysics_new(cart_m, m1, m2, l1_m, l2_m, g);
        this.__wbg_ptr = ret >>> 0;
        WasmPendulumPhysicsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    reset() {
        wasm.wasmpendulumphysics_reset(this.__wbg_ptr);
    }
    /**
     * @returns {any}
     */
    get_state_js() {
        const ret = wasm.wasmpendulumphysics_get_state_js(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get_params_js() {
        const ret = wasm.wasmpendulumphysics_get_params_js(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} dt
     * @param {number} force_override
     * @param {boolean} simulation_mode_is_observing
     * @returns {boolean}
     */
    update_physics_step(dt, force_override, simulation_mode_is_observing) {
        const ret = wasm.wasmpendulumphysics_update_physics_step(this.__wbg_ptr, dt, force_override, simulation_mode_is_observing);
        return ret !== 0;
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_new_405e22f390576ce2 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        arg0[arg1] = arg2;
    };
    imports.wbg.__wbg_warn_bd702e7811e10e4e = function(arg0, arg1) {
        console.warn(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_0;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };
    imports.wbg.__wbindgen_number_new = function(arg0) {
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('physics_engine_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
