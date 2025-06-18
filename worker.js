// This script runs in a separate thread.
// Import TensorFlow.js library within the worker
self.importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js');
// Import the JS glue code for your Wasm physics module
self.importScripts('./pkg_physics/physics_engine.js'); // Path to wasm-pack output

// Enhanced TensorFlow.js Configuration for Performance
async function configureTensorFlowJS() {
    // Enable WebGL backend for better performance
    // Initialize Wasm module (wasm-pack generated)
    // For --target no-modules, wasm_bindgen becomes a global function.
    // self.wasm_bindgen or just wasm_bindgen should work.
    // Pass an object with the module path to satisfy the newer init pattern.
    await wasm_bindgen({ module_or_path: './pkg_physics/physics_engine_bg.wasm' });
    await tf.setBackend('webgl');
    
    // Configure memory management
    tf.env().set('WEBGL_PACK', true);
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
    tf.env().set('WEBGL_RENDER_FLOAT32_CAPABLE', true);
    
    console.log('TensorFlow.js optimized backend:', tf.getBackend());
    console.log('Physics Wasm module loaded.');
    console.log('Memory info:', tf.memory());
}

// Call configuration immediately
configureTensorFlowJS();

// --- Environment Physics (moved to worker) ---
class PendulumPhysics {
    constructor() {
        // Default parameters, Wasm will use these to initialize
        const p = { cart_m: 1.0, m1: 0.1, m2: 0.1, l1_m: 1.0, l2_m: 1.0, g: 9.8 };
        // For --target no-modules, WasmPendulumPhysics becomes a global constructor.
        // self.WasmPendulumPhysics or just WasmPendulumPhysics should work.
        this.wasmInstance = new wasm_bindgen.WasmPendulumPhysics(p.cart_m, p.m1, p.m2, p.l1_m, p.l2_m, p.g);
        this.params = this.wasmInstance.get_params_js(); // Get params from Wasm
        this.state = this.wasmInstance.get_state_js();   // Get initial state from Wasm
        this.actionMax = 5.0; // Reduced from 10.0 to limit max force
        this.maxSteps = 1000;
        this.currentStep = 0;
        // Base reward weights (initial values)
        // dx: Increased significantly to penalize going off-track.
        // dx: Further increased, swing: further reduced to strongly prioritize cart control.
        this.baseRewardWeights = { H: 10, dx: 40.0, effort: 0.01, vel: 0.002, stable: 2.0, swing: 5.0 };

        // Effective reward weights, updated in calculateReward
        this.effectiveRewardWeights = JSON.parse(JSON.stringify(this.baseRewardWeights));
        this.lastRewardComponents = {}; // To store breakdown of reward
        this.lastTerminationReason = 'N/A'; // To store why an episode ended
    }

    reset() {
        this.wasmInstance.reset();
        this.state = this.wasmInstance.get_state_js();
        this.currentStep = 0;
        return this.getStateArray();
        this.lastTerminationReason = 'N/A';
    }

    getStateArray() {
        const s = this.state;
        return [s.cart_x_m, s.cart_x_v_m, Math.cos(s.a1), Math.sin(s.a1), s.a1_v, Math.cos(s.a2), Math.sin(s.a2), s.a2_v];
    }

    getEffectiveRewardWeights() {
        return this.effectiveRewardWeights;
    }

    getRewardComponents() {
        return this.lastRewardComponents || {};
    }

    getTerminationReason() {
        return this.lastTerminationReason || 'N/A';
    }

    step(action, currentSimulationMode = 'TRAINING') { // Parameter renamed for clarity within this function
        // Increment global step counter for annealing logic
        PendulumPhysics.totalSteps = (PendulumPhysics.totalSteps || 0) + 1;
        
        // Call Wasm for physics update
        const dt = 1/60;
        const physics_ok = this.wasmInstance.update_physics_step(dt, action, currentSimulationMode === 'OBSERVING');
        this.state = this.wasmInstance.get_state_js(); // Update JS state from Wasm

        this.currentStep++;
        const next_state = this.getStateArray();
        const reward = this.calculateReward(action);

        let done;
        this.lastTerminationReason = 'Running'; // Default if not done
        if (currentSimulationMode === 'OBSERVING') {
            // In observing mode, the episode effectively doesn't end unless physics break.
            // This allows continuous observation even if cart goes off-track or exceeds typical step limits.
            // The simulation will continue from the current state when switching from TRAINING.
            done = !physics_ok;
            if (done) this.lastTerminationReason = 'Physics Unstable (Observe Mode)';
        } else { // TRAINING or other modes
            // Episode ends if physics unstable, cart off track, or max steps reached.
            if (!physics_ok) {
                done = true; this.lastTerminationReason = 'Physics Unstable';
            } else if (Math.abs(this.state.cart_x_m) > 4.5) {
                done = true; this.lastTerminationReason = 'Cart Out of Bounds';
            } else if (this.currentStep >= this.maxSteps) {
                done = true; this.lastTerminationReason = 'Max Episode Steps Reached';
            } else {
                done = false;
            }
        }
        return { next_state, reward, done };
    }

    calculateReward(action) {
        const { a1, a2, a1_v, a2_v, cart_x_m } = this.state;
        const { l1_m, l2_m } = this.params;

        // === 0. Sanity-check ===
        const varsToCheck = [a1, a2, a1_v, a2_v, cart_x_m, action];
        if (varsToCheck.some(v => !isFinite(v) || isNaN(v))) {
            console.warn('NaN/Inf in reward inputs', varsToCheck);
            return -10; // harsh penalty for invalid state
        }

        // === 1. Keep a global step counter so we can anneal shaping terms ===
        if (typeof PendulumPhysics.totalSteps === 'undefined') {
            PendulumPhysics.totalSteps = 0;
        }
        const globalStep = PendulumPhysics.totalSteps;

        // === 2. Core features ===
        const cos_a1 = Math.cos(a1);
        const cos_a2 = Math.cos(a2);

        // 2.1  Height of pole-2 tip (normalised 0→2).
        //      Upright → high reward; hanging ↓ low reward.
        const heightTip2 =
            (l1_m * (1 - cos_a1) + l2_m * (1 - cos_a2)) / (l1_m + l2_m); // 0→2

        // 2.2  Cart displacement (centre of rail at 0).
        // Adjusted TRACK_HALF to match episode termination condition (abs(cart_x_m) > 4.5)
        const TRACK_HALF = 4.5; 
        const cartPenalty = (cart_x_m / TRACK_HALF) ** 2; // Quadratic penalty: 0 @ centre, 1 @ rail end

        // 2.3  Effort penalty – square of commanded force, scaled to [0,1].
        const effort = (action / this.actionMax) ** 2; // ∈ [0,1]

        // 2.4  Residual wobble (angular velocity).
        const velPenalty = (a1_v * a1_v + a2_v * a2_v);

        // 2.5  Stability bonus (tiny but crisp signal once balanced).
        const STABLE_ANGLE = 12 * Math.PI / 180; // 12°
        const STABLE_CART  = 0.2; // m
        const isStable =
            (Math.abs(a1 - Math.PI) < STABLE_ANGLE) &&
            (Math.abs(a2 - Math.PI) < STABLE_ANGLE) &&
            (Math.abs(cart_x_m)    < STABLE_CART) ? 1 : 0;

        // === 3. Coefficients w/ annealing for height shaping ===
        // Calculate annealed weights for this step based on baseRewardWeights
        let currentH = this.baseRewardWeights.H;
        const ANNEAL_START   = 500000; // Start annealing H much later
        const ANNEAL_PERIOD  = 100000; // Anneal H over a longer period
        const DECAY          = 0.95;   // Slower decay rate for H

        // Anneal height shaping every ANNEAL_PERIOD env steps (min 1).
        if (globalStep > ANNEAL_START) {
            const numAnnealingPeriods = Math.floor((globalStep - ANNEAL_START) / ANNEAL_PERIOD);
            currentH = this.baseRewardWeights.H * Math.pow(DECAY, numAnnealingPeriods);
            currentH = Math.max(1.0, currentH); // Ensure H doesn't go too low
        }

        let currentVelWeight = this.baseRewardWeights.vel; // Base velocity damping
        // Increase velocity damping if pendulums are mostly upright
        if (cos_a1 < -0.8 || cos_a2 < -0.8) { // cos(angle) is negative when up
            currentVelWeight += 0.05; 
        }

        // Update effectiveRewardWeights for logging/external access
        this.effectiveRewardWeights.H = currentH;
        this.effectiveRewardWeights.vel = currentVelWeight;
        this.effectiveRewardWeights.dx = this.baseRewardWeights.dx;
        this.effectiveRewardWeights.effort = this.baseRewardWeights.effort;
        this.effectiveRewardWeights.stable = this.baseRewardWeights.stable;
        this.effectiveRewardWeights.swing = this.baseRewardWeights.swing;

        // === 3.5 Swing-up assistance reward ===
        // Reward for actions that contribute to swinging the pendulums up
        // action is normalized [-1, 1]. a1_v, a2_v are angular velocities.
        // cos(a1) > 0.1 means pendulum 1 is in the lower half (a1=0 is down)
        let swing_assist_reward = 0;
        if (cos_a1 > 0.1) { swing_assist_reward += this.baseRewardWeights.swing * action * a1_v * cos_a1; }
        if (cos_a2 > 0.1) { swing_assist_reward += this.baseRewardWeights.swing * action * a2_v * cos_a2; }


        // === 4. Compose reward ===
        const rH = this.effectiveRewardWeights.H * heightTip2;
        const rDx = -this.effectiveRewardWeights.dx * cartPenalty;
        const rEffort = -this.effectiveRewardWeights.effort * effort;
        const rVel = -this.effectiveRewardWeights.vel * velPenalty;
        const rStable = this.effectiveRewardWeights.stable * isStable;
        // swing_assist_reward is already calculated with its weight

        this.lastRewardComponents = {
            rH, rDx, rEffort, rVel, rStable, rSwing: swing_assist_reward
        };

        let r = rH + rDx + rEffort + rVel + rStable + swing_assist_reward;




        // Clip to keep numerical range tame.
        r = Math.max(-50, Math.min(50, r));
        return r;
    }

    // updatePhysics method is now implicitly handled by Wasm.
    // The JS `step` method calls Wasm's `update_physics_step` and then updates `this.state`.
    // mat_inv and multiplyMatrixVector are now internal to the Wasm module.
}

// Custom Layer for Scaling
class ScaleLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.scaleFactor = config.scaleFactor;
    }

    computeOutputShape(inputShape) {
        return inputShape; // Output shape is the same as input shape
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;
            return tf.mul(input, tf.scalar(this.scaleFactor));
        });
    }

    static get className() {
        return 'ScaleLayer'; // Important for serialization
    }
}

class SACAgent {
    constructor() {
        this.stateSize = 8;
        this.actionSize = 1;
        this.actionBounds = 1.0; // Actions in [-1, 1]
        
        // IMPROVED SAC Hyperparameters for better learning
        this.actorLr = 0.0005; // Slightly higher than default
        this.criticLr = 0.0005; // Slightly higher than default
        this.initialAlpha = 0.2; // Starting entropy coefficient
        this.alpha = this.initialAlpha;
        this.gamma = 0.99;
        this.tau = 0.005; // Reduced from 0.01 for more stable target updates
        this.batchSize = 128; // Reduced from 256 for more frequent updates
        this.bufferSize = 100000;
        this.warmupSteps = 5000; // Increased for more initial random exploration

        // Automatic entropy tuning
        this.alphaLr = 0.0003;
        this.logAlpha = tf.variable(tf.scalar(Math.log(this.initialAlpha)));
        this.alphaOptimizer = tf.train.adam(this.alphaLr);
        this.targetEntropy = -this.actionSize; // heuristic target
        
        // Experience replay
        this.replayBuffer = [];
        this.isReady = false;
        
        // State normalization
        this.stateRunningMean = new Array(this.stateSize).fill(0);
        this.stateRunningVar = new Array(this.stateSize).fill(1);
        // For diagnostics
        this.lastActorMean = null;
        this.lastActorLogStd = null;
        this.lastQ1Value = null;
        this.lastQ2Value = null;
        this.stateCount = 0;
        
        this.init();
    }

    async init() {
        // Register custom layer
        tf.serialization.registerClass(ScaleLayer);

        // Build networks
        this.actor = this.buildActor();
        // Log actor summary to see layers
        // this.actor.summary(); 

        this.critic1 = this.buildCritic();
        this.critic2 = this.buildCritic(); // Twin Q-networks
        this.targetCritic1 = this.buildCritic();
        this.targetCritic2 = this.buildCritic();

        // Optimizers
        this.actorOptimizer = tf.train.adam(this.actorLr, undefined, undefined, undefined, 1.0); // Added clipnorm
        this.critic1Optimizer = tf.train.adam(this.criticLr, undefined, undefined, undefined, 1.0); // Added clipnorm
        this.critic2Optimizer = tf.train.adam(this.criticLr, undefined, undefined, undefined, 1.0); // Added clipnorm

        // Copy weights to target networks
        this.updateTargetNetworks(1.0);
        this.isReady = true;
        
        console.log('SAC Agent initialized with improved architecture');
    }

    buildActor() {
        const input = tf.input({shape: [this.stateSize]});
        
        // Larger network with proper initialization
        let x = tf.layers.dense({
            units: 256, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(input);
        
        x = tf.layers.layerNormalization().apply(x);
        
        x = tf.layers.dense({
            units: 256, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(x);
        
        x = tf.layers.layerNormalization().apply(x);
        
        // Mean and log_std outputs for policy
        const meanOutput = tf.layers.dense({
            units: this.actionSize, 
            activation: 'tanh', // Changed to tanh
            kernelInitializer: 'truncatedNormal',
            name: 'policy_mean'
        }).apply(x);
        
        const logStdOutput = tf.layers.dense({
            units: this.actionSize, 
            activation: 'tanh', // Changed to tanh
            kernelInitializer: 'truncatedNormal',
            name: 'policy_log_std'
        }).apply(x);
        
        // Scale meanOutput by actionBounds
        const scaledMeanOutput = new ScaleLayer({ scaleFactor: this.actionBounds, name: 'scale_action_mean' })
            .apply(meanOutput);
        
        return tf.model({inputs: input, outputs: [scaledMeanOutput, logStdOutput]});
    }

    buildCritic() {
        const stateInput = tf.input({shape: [this.stateSize], name: 'state'});
        const actionInput = tf.input({shape: [this.actionSize], name: 'action'});
        
        // State processing
        let stateFeatures = tf.layers.dense({
            units: 256, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(stateInput);
        
        // Action processing
        let actionFeatures = tf.layers.dense({
            units: 256, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(actionInput);
        
        // Concatenate state and action features
        const concat = tf.layers.concatenate().apply([stateFeatures, actionFeatures]);
        
        let x = tf.layers.layerNormalization().apply(concat);
        
        x = tf.layers.dense({
            units: 256, 
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }).apply(x);
        
        x = tf.layers.layerNormalization().apply(x);
        
        const output = tf.layers.dense({
            units: 1,
            kernelInitializer: 'truncatedNormal'
        }).apply(x);
        
        return tf.model({inputs: [stateInput, actionInput], outputs: output});
    }
    
    normalizeState(state) {
        // Enhanced online normalization with robust numerical stability
        const normalized = [];
        
        // Validate input state first
        if (!Array.isArray(state) || state.length !== this.stateSize) {
            console.warn('Invalid state in normalizeState:', state);
            return new Array(this.stateSize).fill(0); // Return safe default
        }
        
        for (let i = 0; i < state.length; i++) {
            // Check for invalid input values
            if (!isFinite(state[i]) || isNaN(state[i])) {
                console.warn(`Invalid state value at index ${i}:`, state[i]);
                normalized[i] = 0; // Use safe default
                continue;
            }
            
            // Clamp extreme values before processing
            const clampedValue = Math.max(-1000, Math.min(1000, state[i]));
            
            // Update running statistics with numerical stability
            this.stateCount++;
            const count = Math.min(this.stateCount, 10000); // Cap for numerical stability
            
            const delta = clampedValue - this.stateRunningMean[i];
            this.stateRunningMean[i] += delta / count;
            
            // More stable variance calculation
            const delta2 = clampedValue - this.stateRunningMean[i];
            this.stateRunningVar[i] = (this.stateRunningVar[i] * (count - 1) + delta * delta2) / count;
            
            // Ensure variance doesn't become too small or invalid
            this.stateRunningVar[i] = Math.max(this.stateRunningVar[i], 1e-6);
            
            // Check for invalid statistics
            if (!isFinite(this.stateRunningMean[i]) || isNaN(this.stateRunningMean[i])) {
                console.warn(`Invalid running mean at index ${i}, resetting`);
                this.stateRunningMean[i] = 0;
            }
            
            if (!isFinite(this.stateRunningVar[i]) || isNaN(this.stateRunningVar[i])) {
                console.warn(`Invalid running variance at index ${i}, resetting`);
                this.stateRunningVar[i] = 1;
            }
            
            // Normalize with robust division
            const std = Math.sqrt(this.stateRunningVar[i]);
            const normalizedValue = (clampedValue - this.stateRunningMean[i]) / Math.max(std, 1e-6);
            
            // Final clipping and validation
            normalized[i] = Math.max(-10, Math.min(10, normalizedValue));
            
            // Final safety check
            if (!isFinite(normalized[i]) || isNaN(normalized[i])) {
                console.warn(`Final normalized value invalid at index ${i}, using 0`);
                normalized[i] = 0;
            }
        }
        
        return normalized;
    }
    
    sampleAction(mean, logStd, deterministic = false) {
        return tf.tidy(() => {
            try {
                if (deterministic) {
                    // 'mean' is already scaled and tanh'd by the actor network's output layer
                    // So, for deterministic action, 'mean' itself is the action.
                    const action = mean; 
                    return {action, logProb: tf.zeros([action.shape[0], 1])};
                }
                
                // logStd is now the direct output of a tanh layer, in [-1, 1]
                // Scale it to a more appropriate range for log_std, e.g., [-5, 2]
                const LOG_STD_MAX = 2.0;
                const LOG_STD_MIN = -5.0; // Tighter range
                // scaledLogStd = logStd_from_actor_tanh * 3.5 - 1.5
                const scaledLogStd = logStd.mul(0.5 * (LOG_STD_MAX - LOG_STD_MIN)).add(0.5 * (LOG_STD_MAX + LOG_STD_MIN));

                const clippedLogStd = tf.clipByValue(scaledLogStd, LOG_STD_MIN, LOG_STD_MAX);
                const std = tf.exp(clippedLogStd);

                // Add small epsilon to prevent numerical issues
                const epsilon = 1e-6;
                const safeStd = tf.maximum(std, epsilon);
                
                const noise = tf.randomNormal(mean.shape);
                // 'mean' is already scaled by actionBounds and passed through tanh in the actor.
                // For sampling, we add noise to this pre-squashed mean, then re-apply tanh if necessary,
                // but since 'mean' is already the output of a tanh, we can directly use it.
                // The reparameterization trick: action = mean + std * noise
                const actionSample = mean.add(safeStd.mul(noise));
                const action = tf.tanh(actionSample); // Squash the sum to ensure it's within bounds for logProb calc
                
                // More robust log probability calculation
                const normalizedAction = actionSample.sub(mean).div(safeStd); // Use actionSample before final tanh for logProb
                
                // Compute log probability with better numerical stability
                const logProbNormal = tf.sum(
                    tf.mul(-0.5, tf.square(normalizedAction))
                    .sub(tf.scalar(0.5 * Math.log(2 * Math.PI)))
                    .sub(clippedLogStd),
                    -1,
                    true
                );
                
                // Tanh correction with numerical stability
                const actionSquared = tf.square(action);
                const oneMinusActionSq = tf.sub(1, actionSquared);
                const safeOneMinusActionSq = tf.maximum(oneMinusActionSq, epsilon);
                
                const tanhCorrection = tf.sum(
                    tf.log(safeOneMinusActionSq),
                    -1,
                    true
                );
                
                const logProb = logProbNormal.add(tanhCorrection);
                
                // Final validation and clipping of log probabilities
                const clippedLogProb = tf.clipByValue(logProb, -50, 50);
                
                return {action, logProb: clippedLogProb};
                
            } catch (error) {
                console.error('Error in sampleAction:', error);
                // Return safe fallback values
                const fallbackAction = tf.randomUniform(mean.shape, -1, 1);
                const fallbackLogProb = tf.fill([mean.shape[0], 1], -1.0);
                return {action: fallbackAction, logProb: fallbackLogProb};
            }
        });
    }
    
    chooseAction(state, deterministic = false) {
        try {
            return tf.tidy(() => {
                // Validate input state
                if (!Array.isArray(state) || state.length !== this.stateSize) {
                    console.warn('Invalid state in chooseAction:', state);
                    return 0; // Return safe default action
                }
                
                const normalizedState = this.normalizeState(state);
                
                // Additional validation of normalized state
                for (let i = 0; i < normalizedState.length; i++) {
                    if (!isFinite(normalizedState[i]) || isNaN(normalizedState[i])) {
                        console.warn('Invalid normalized state in chooseAction:', normalizedState);
                        return 0; // Return safe default action
                    }
                }
                
                const stateTensor = tf.tensor2d([normalizedState]);
                const [mean, logStd] = this.actor.apply(stateTensor);

                // Store for diagnostics
                this.lastActorMean = mean.dataSync()[0]; // Assuming actionSize is 1
                this.lastActorLogStd = logStd.dataSync()[0]; // Assuming actionSize is 1
                
                // Validate network outputs
                const meanData = mean.dataSync();
                const logStdData = logStd.dataSync();
                
                if (meanData.some(x => !isFinite(x) || isNaN(x)) || 
                    logStdData.some(x => !isFinite(x) || isNaN(x))) {
                    console.warn('Invalid network outputs in chooseAction');
                    return 0; // Return safe default action
                }
                
                const {action, logProb: _logProb} = this.sampleAction(mean, logStd, deterministic);

                // Get Q-values for the chosen action (for diagnostics)
                const actionTensor = action; // action is already a tensor
                this.lastQ1Value = this.critic1.apply([stateTensor, actionTensor]).dataSync()[0];
                this.lastQ2Value = this.critic2.apply([stateTensor, actionTensor]).dataSync()[0];

                const actionValue = action.dataSync()[0];
                
                // Final validation of action
                if (!isFinite(actionValue) || isNaN(actionValue)) {
                    console.warn('Invalid action generated:', actionValue);
                    return 0; // Return safe default action
                }
                
                // Clamp action to valid range
                return Math.max(-1, Math.min(1, actionValue));
            });
        } catch (error) {
            console.error('Error in chooseAction:', error);
            return 0; // Return safe default action
        }
    }

    getPolicyDiagnostics() {
        return {
            actorMean: this.lastActorMean,
            actorLogStd: this.lastActorLogStd,
            q1Value: this.lastQ1Value,
            q2Value: this.lastQ2Value,
        };
    }
    
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push({
            state: this.normalizeState(state), 
            action, 
            reward, 
            nextState: this.normalizeState(nextState), 
            done
        });
        
        if (this.replayBuffer.length > this.bufferSize) {
            this.replayBuffer.shift();
        }
    }
    
    train() {
        if (this.replayBuffer.length < this.batchSize) return null;

        // Sample batch
        const batch = this.sampleBatch();
        
        return tf.tidy(() => {
            const states = tf.tensor2d(batch.states);
            const actions = tf.tensor2d(batch.actions);
            const rewards = tf.tensor2d(batch.rewards);
            const nextStates = tf.tensor2d(batch.nextStates);
            const dones = tf.tensor2d(batch.dones);

            // Update critics with memory management
            const criticLossTensor = this.updateCritics(states, actions, rewards, nextStates, dones);
            
            // FIXED: Update actor every training step for better learning
            const actorUpdateResult = this.updateActor(states); // Returns {loss, avgLogProb, logProbs} tensors
            const alphaLossValue = this.updateAlpha(actorUpdateResult.logProbs);
            actorUpdateResult.logProbs.dispose();
            
            // Soft update target networks
            this.updateTargetNetworks(this.tau);
            
            // Extract values before tensors are disposed
            const criticLossValue = criticLossTensor.dataSync()[0];
            const actorLossValue = actorUpdateResult && actorUpdateResult.loss ? actorUpdateResult.loss.dataSync()[0] : null;
            const avgLogProbValue = actorUpdateResult && actorUpdateResult.avgLogProb ? actorUpdateResult.avgLogProb.dataSync()[0] : null;
            
            // DEBUG: Log calculated losses in worker
            console.log('[Worker] SACAgent.train() losses:', { criticLoss: criticLossValue, actorLoss: actorLossValue, alphaLoss: alphaLossValue, avgLogProb: avgLogProbValue, alpha: this.alpha });

            return {
                criticLoss: criticLossValue,
                actorLoss: actorLossValue,
                alphaLoss: alphaLossValue,
                avgLogProb: avgLogProbValue,
                alpha: this.alpha
            };
        });

    }
    
    sampleBatch() {
        const batch = {states: [], actions: [], rewards: [], nextStates: [], dones: []};
        
        // Validate replayBuffer before sampling
        if (!this.replayBuffer || this.replayBuffer.length === 0) {
            console.warn("Replay buffer is empty or invalid during sampleBatch.");
            // Return an empty batch or handle error appropriately
            return batch; 
        }

        for (let i = 0; i < this.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.replayBuffer.length);
            const experience = this.replayBuffer[idx];
            
            if (!experience) continue; // Skip if experience is undefined
            batch.states.push(experience.state);
            batch.actions.push([experience.action]);
            batch.rewards.push([experience.reward]);
            batch.nextStates.push(experience.nextState);
            batch.dones.push([experience.done ? 1 : 0]);
        }
        
        return batch;
    }
    
    updateCritics(states, actions, rewards, nextStates, dones) {
        // Target Q-values calculation
        const targetQ = tf.tidy(() => { // This tidy keeps the returned targetQ tensor
            const [nextMeans, nextLogStds] = this.actor.apply(nextStates);
            const {action: nextActions, logProb: nextLogProbs} = this.sampleAction(nextMeans, nextLogStds);
            
            const targetQ1 = this.targetCritic1.apply([nextStates, nextActions]);
            const targetQ2 = this.targetCritic2.apply([nextStates, nextActions]);
            const minTargetQ = tf.minimum(targetQ1, targetQ2);
            
            const entropyTerm = tf.mul(this.alpha, nextLogProbs);
            const targetValue = minTargetQ.sub(entropyTerm);
            
            return rewards.add(
                dones.mul(-1).add(1).mul(this.gamma).mul(targetValue)
            );
        });
    
        // Update critic 1
        const critic1LossFn = () => {
            const currentQ1 = this.critic1.apply([states, actions]);
            return tf.mean(tf.square(targetQ.sub(currentQ1)));
        };
    
        const critic1Result = tf.variableGrads(critic1LossFn, this.critic1.trainableWeights.map(v => v.val));
        this.critic1Optimizer.applyGradients(critic1Result.grads);
        tf.dispose(critic1Result.grads); // Explicitly dispose gradients
    
        // Update critic 2
        const critic2LossFn = () => {
            const currentQ2 = this.critic2.apply([states, actions]);
            return tf.mean(tf.square(targetQ.sub(currentQ2)));
        };
    
        const critic2Result = tf.variableGrads(critic2LossFn, this.critic2.trainableWeights.map(v => v.val));
        this.critic2Optimizer.applyGradients(critic2Result.grads);
        tf.dispose(critic2Result.grads); // Explicitly dispose gradients
    
        // Return the loss value from critic1 that was used for gradient calculation.
        // critic1Result.value is kept by tf.variableGrads.
        return critic1Result.value;
    }
    
    updateActor(states) {
        let lossTensor, avgLogProbTensor, logProbsOut;

        const grads = tf.variableGrads(() => {
            const [means, logStds] = this.actor.apply(states);
            const {action: newActions, logProb: logProbs} = this.sampleAction(means, logStds); // logProbs is used here
            
            const q1 = this.critic1.apply([states, newActions]);
            const q2 = this.critic2.apply([states, newActions]);
            const minQ = tf.minimum(q1, q2);

            // SAC actor loss: maximize Q - alpha * entropy
            const entropyTerm = tf.mul(this.alpha, logProbs);
            const currentLoss = tf.mean(entropyTerm.sub(minQ));

            // lossTensor and avgLogProbTensor were assigned here.
            // currentLoss is the return value, so it will be grads.value.
            // logProbs (and tf.mean(logProbs)) created here might be tidied by variableGrads.
            return currentLoss; // Return loss for gradient calculation
        }, this.actor.trainableWeights.map(v => v.val));

        this.actorOptimizer.applyGradients(grads.grads);
        tf.dispose(grads.grads); // Explicitly dispose gradients

        // Re-evaluate policy to get avgLogProb safely, ensuring tensors are not disposed.
        // This is slightly less efficient but safer for tensor lifecycle.
        const [finalMeans, finalLogStds] = this.actor.apply(states);
        const {logProb: finalLogProbs} = this.sampleAction(finalMeans, finalLogStds, false);
        avgLogProbTensor = tf.mean(finalLogProbs); // This tensor is created now.
        logProbsOut = finalLogProbs;

        // grads.value is the loss tensor computed by the function passed to variableGrads, and it's kept.
        lossTensor = grads.value;

        return { loss: lossTensor, avgLogProb: avgLogProbTensor, logProbs: logProbsOut }; // These tensors should now be valid
    }

    updateAlpha(logProbs) {
        const alphaGrads = tf.variableGrads(() => {
            return tf.tidy(() => {
                const diff = logProbs.add(this.targetEntropy);
                return tf.mean(this.logAlpha.mul(diff));
            });
        }, [this.logAlpha]);

        this.alphaOptimizer.applyGradients(alphaGrads.grads);
        const lossValue = alphaGrads.value.dataSync()[0];
        tf.dispose(alphaGrads.grads);
        alphaGrads.value.dispose();
        this.alpha = Math.exp(this.logAlpha.dataSync()[0]);
        return lossValue;
    }

    updateTargetNetworks(tau) {
        // Soft update target critic networks
        const updateTarget = (target, source) => {
            const targetWeights = target.getWeights();
            const sourceWeights = source.getWeights();
            const newWeights = sourceWeights.map((w, i) => 
                tf.tidy(() => w.mul(tau).add(targetWeights[i].mul(1 - tau)))
            );
            target.setWeights(newWeights);
        };

        updateTarget(this.targetCritic1, this.critic1);
        updateTarget(this.targetCritic2, this.critic2);
    }
}

// --- Worker Globals ---
let allowRender = true; // main-thread can disable to save bandwidth during training
let simulationMode = 'IDLE'; // 'IDLE', 'TRAINING', 'OBSERVING'
let isPaused = false; // General pause flag, true if simulationMode was TRAINING or OBSERVING and pause_simulation was called
let isObservingPolicyWhileTrainingPaused = false; // Specific flag for when training is "paused" but we're watching the policy
let previousSimulationMode = 'IDLE';
let agent = null;
let physics = null;
// let isTraining = false; // Replaced by simulationMode and isTrainingInternal
let userSetStepsPerFrame = 1; // Speed set by the user via slider
let totalSteps = 0;
let episode = 0;
let totalReward = 0;
let state = null;
let latestAction = 0;
let lastTrainStep = 0; // Tracks totalSteps at last training event
let episodeRewards = [];
    let lastPolicyDiagnostics = {}; // To store agent's internal thoughts
let currentSimStepsPerFrame = 1; // Actual steps used in loop, adjusted for OBSERVE mode
let bestReward = -Infinity;
let allowRenderBeforeTrainingPause = false; // To restore render state when resuming training
let lastTrainingLosses = null; // To store results from agent.train()
let lastSpsCheckTime = 0;
let stepsSinceLastSpsCheck = 0;
// const WARMUP_STEPS = 1000; // This is superseded by AGENT_WARMUP_STEPS for agent logic
const TRAIN_FREQUENCY = 1; // Train after every block of userSetStepsPerFrame steps if slider is >=1x
const MAX_EPISODE_STEPS = 2000; // Longer episodes
const AGENT_WARMUP_STEPS = 5000; // Should match agent.warmupSteps
// DEBUGGING HELPERS - Add comprehensive NaN detection
function isValidNumber(value) {
    return typeof value === 'number' && isFinite(value) && !isNaN(value);
}

function validateState(state, context = 'unknown') {
    if (!Array.isArray(state)) {
        console.error(`Invalid state format in ${context}:`, state);
        return false;
    }
    
    for (let i = 0; i < state.length; i++) {
        if (!isValidNumber(state[i])) {
            console.error(`NaN/Infinite value in state[${i}] in ${context}:`, state[i]);
            return false;
        }
    }
    return true;
}

function validateReward(reward, context = 'unknown') {
    if (!isValidNumber(reward)) {
        console.error(`Invalid reward in ${context}:`, reward);
        return false;
    }
    return true;
}

function init() {
    agent = new SACAgent();
    physics = new PendulumPhysics();
    physics.maxSteps = MAX_EPISODE_STEPS; // Update physics
    state = physics.reset();
    simulationMode = 'IDLE';
    isPaused = false;
    isObservingPolicyWhileTrainingPaused = false;
    
    // Validate initial state
    if (!validateState(state, 'init')) {
        console.error('Invalid initial state, resetting...');
        state = physics.reset();
    }
    
    latestAction = 0;
    totalSteps = 0;
    episode = 0;
    totalReward = 0;
    lastTrainStep = 0;
    episodeRewards = [];
    bestReward = -Infinity;
    lastSpsCheckTime = performance.now();
    stepsSinceLastSpsCheck = 0;
    console.log('Initializing robust SAC agent for double pendulum...');
}

function simulationStep() {
    if (!agent || !agent.isReady || simulationMode === 'IDLE' || (isPaused && !isObservingPolicyWhileTrainingPaused)) return;

    const currentStateForAction = physics.getStateArray(); // State before stepping
    const currentState = currentStateForAction; // Alias for clarity in existing validation
    
    // Validate current state
    if (!validateState(currentState, 'simulationStep-current')) {
        console.warn('Invalid current state detected, resetting episode');
        state = physics.reset();
        return;
    }
    
    let actionToTake;
    const isDeterministicAction = simulationMode === 'OBSERVING' || isObservingPolicyWhileTrainingPaused;

    if (simulationMode === 'TRAINING' && !isObservingPolicyWhileTrainingPaused) { // True training step
        if (totalSteps < AGENT_WARMUP_STEPS) { // Use agent's warmup steps
            actionToTake = (Math.random() * 2 - 1);
        } else {
            actionToTake = agent.chooseAction(currentStateForAction, false); // Stochastic
        }
    } else { // OBSERVING mode or observing policy while training is "paused"
        actionToTake = agent.chooseAction(currentStateForAction, isDeterministicAction); // Deterministic
    }
    latestAction = actionToTake; // Store the chosen action (normalized)

    if (!isValidNumber(actionToTake)) {
        console.warn(`Invalid action ${actionToTake} in ${simulationMode}, using fallback.`);
        actionToTake = 0; // Fallback normalized action
        latestAction = 0;
    }
    lastPolicyDiagnostics = agent.getPolicyDiagnostics(); // Get diagnostics after action is chosen
    
    const scaledAction = actionToTake * physics.actionMax;
    const { next_state, reward, done } = physics.step(scaledAction, simulationMode); // Pass the global simulationMode

    // Comprehensive validation
    if (!validateState(next_state, 'simulationStep-next')) {
        console.warn('Invalid next state, resetting episode');
        state = physics.reset();
        return;
    }

    if (!validateReward(reward, 'simulationStep')) {
        console.warn('Invalid reward detected, using fallback reward');
        const fallbackReward = -1.0; // Small negative reward for invalid state
        totalReward += fallbackReward;
        if (!isValidNumber(totalReward)) {
            totalReward = -100.0; // Reset to reasonable value
        }
    } else {
        totalReward += reward;
    }

    // Store experience and increment totalSteps only if in actual TRAINING mode (not observing paused policy)
    if (simulationMode === 'TRAINING' && !isObservingPolicyWhileTrainingPaused) {
        totalSteps++; // Only count steps that contribute to training
        if (validateState(currentStateForAction, 'experience-current') && 
            validateState(next_state, 'experience-next') && 
            isValidNumber(actionToTake) && 
            isValidNumber(reward)) {
            agent.remember(currentStateForAction, actionToTake, reward, next_state, done);
        } else {
            console.warn('Skipping experience storage due to invalid values');
        }
    }
    
    state = next_state;
    
    // Get next action
    // latestAction is already set based on mode for the current step's outcome.

    if (done) {
        // Validate episode reward before logging
        if (!isValidNumber(totalReward)) {
            console.warn('Invalid total reward at episode end:', totalReward);
            totalReward = -100.0; // Use reasonable fallback
        }
        
        episodeRewards.push(totalReward);
        
        if (totalReward > bestReward) {
            bestReward = totalReward;
        }
        
        const avgReward = episodeRewards.length >= 10 ? 
            episodeRewards.slice(-10).reduce((a, b) => a + b) / 10 : totalReward;
        
        // Validate avgReward
        const validAvgReward = isValidNumber(avgReward) ? avgReward : totalReward;
        
        self.postMessage({ 
            type: 'episode_done', 
            payload: { 
                episode, 
                totalReward, 
                bestReward,
                avgReward: validAvgReward,
                totalSteps: simulationMode === 'TRAINING' ? totalSteps : undefined,
                mode: isObservingPolicyWhileTrainingPaused ? 'TRAINING_PAUSED_OBSERVING' : simulationMode,
                bufferSize: agent.replayBuffer.length
            },
            // Add more detailed snapshot data
            trainingLosses: lastTrainingLosses, 
            lastStepRewardComponents: physics.getRewardComponents(),
            // Send the action and diagnostics for the step that ended the episode
            lastAction: latestAction, 
            policyDiagnostics: lastPolicyDiagnostics,
            terminationReason: physics.getTerminationReason(),
            agentConfig: {
                alpha: agent.alpha,
                actorLr: agent.actorLr,
                criticLr: agent.criticLr,
                batchSize: agent.batchSize,
                tau: agent.tau,
                gamma: agent.gamma,
                bufferSize: agent.bufferSize, // Agent's configured max buffer
                warmupSteps: AGENT_WARMUP_STEPS, // Use the agent's actual warmup steps
                trainFrequency: TRAIN_FREQUENCY, // Use the global constant
                stateSize: agent.stateSize,
                actionSize: agent.actionSize,
                alphaLr: agent.alphaLr,
                targetEntropy: agent.targetEntropy,
                stateNormalizationMean: agent.stateRunningMean.map(v => parseFloat(v.toFixed(4))), // Add normalization stats
                stateNormalizationVar: agent.stateRunningVar.map(v => parseFloat(v.toFixed(4)))   // Add normalization stats
            },
            physicsRewardConfig: physics.getEffectiveRewardWeights(),
            currentSpeed: currentSimStepsPerFrame
        });
        
        episode++;
        totalReward = 0;
        state = physics.reset();
        
        // Validate reset state
        if (!validateState(state, 'episode-reset')) {
            console.error('Invalid state after reset, forcing new reset');
            physics = new PendulumPhysics(); // Create new physics instance
            state = physics.reset();
        }
        
        // Reset action for the new episode (will be chosen at start of next simulationStep)
        if ((simulationMode !== 'TRAINING' || isObservingPolicyWhileTrainingPaused) || totalSteps < AGENT_WARMUP_STEPS) {
            latestAction = 0;
        }
        
        // Memory management
        if (episode % 100 === 0) {
            if (typeof tf !== 'undefined' && tf.memory) {
                console.log(`Episode ${episode}, Memory: ${tf.memory().numTensors} tensors`);
                if (typeof gc !== 'undefined') gc();
            }
        }
    }
}

function runSimulationLoop() {
    const now = performance.now(); // Get time at the start of the loop invocation

    if (simulationMode === 'IDLE' || (isPaused && !isObservingPolicyWhileTrainingPaused)) {
        // If simulation is not actively running, report 0 SPS if enough time has passed or if there were pending steps
        if (stepsSinceLastSpsCheck > 0 || (now - lastSpsCheckTime > 1000)) { // Check if an update is due
            self.postMessage({ type: 'sps_update', payload: { sps: 0 } });
            lastSpsCheckTime = now;
            stepsSinceLastSpsCheck = 0;
        }
        return; // Stop the loop
    }

    const isEffectivelyObserving = simulationMode === 'OBSERVING' || isObservingPolicyWhileTrainingPaused;
    currentSimStepsPerFrame = isEffectivelyObserving ? 1 : userSetStepsPerFrame;

    for (let i = 0; i < currentSimStepsPerFrame; i++) {
        simulationStep();
    }

    // SPS Counter
    if (simulationMode === 'TRAINING' && !isObservingPolicyWhileTrainingPaused) {
        stepsSinceLastSpsCheck += currentSimStepsPerFrame;
        if (now - lastSpsCheckTime >= 990) { // Update SPS roughly every second
            const elapsedSeconds = (now - lastSpsCheckTime) / 1000;
            const sps = elapsedSeconds > 0 ? stepsSinceLastSpsCheck / elapsedSeconds : 0;
            self.postMessage({ type: 'sps_update', payload: { sps: sps.toFixed(0) } });
            lastSpsCheckTime = now;
            stepsSinceLastSpsCheck = 0;
        }
    } else if (isEffectivelyObserving) {
        // For observing mode, SPS is roughly 60 due to setTimeout aiming for ~16.6ms per frame (and 1 step per frame)
        if (now - lastSpsCheckTime >= 990) {
            self.postMessage({ type: 'sps_update', payload: { sps: (1 * 60).toFixed(0) } }); // Approx 60 SPS
            lastSpsCheckTime = now;
        }
    }
    
    // Train the agent with enhanced error handling
    if (simulationMode === 'TRAINING' && !isObservingPolicyWhileTrainingPaused && // Only train if truly training
        totalSteps > AGENT_WARMUP_STEPS && // Use agent's warmup steps
        (totalSteps - lastTrainStep) >= TRAIN_FREQUENCY && 
        agent.replayBuffer.length >= agent.batchSize) {
        
        try {
            const losses = agent.train();
            lastTrainStep = totalSteps;
            lastTrainingLosses = losses; // Store the latest losses
            
            // Validate training losses
            if (losses) {
                if (!isValidNumber(losses.criticLoss)) {
                    console.warn('Invalid critic loss detected:', losses.criticLoss);
                }
                if (losses.actorLoss && !isValidNumber(losses.actorLoss)) {
                    console.warn('Invalid actor loss detected:', losses.actorLoss);
                }
                
                // Optional: Log training progress (ensure avgLogProb is checked if present)
                if (totalSteps % 1000 === 0) {
                    let logMsg = `Step ${totalSteps}: Critic Loss = ${losses.criticLoss?.toFixed(4)}, Actor Loss = ${losses.actorLoss?.toFixed(4)}`;
                    if (losses.avgLogProb !== undefined && isValidNumber(losses.avgLogProb)) {
                        logMsg += `, AvgLogProb = ${losses.avgLogProb.toFixed(4)}`;
                    }
                    console.log(logMsg);
                }
            }
        } catch (error) {
            console.error('Error during training at step', totalSteps, ':', error);
            lastTrainingLosses = null; // Clear on error
            // Skip this training step but continue simulation
        }
    }

    // Send render data with validation
    if (physics && physics.state) {
        if (allowRender && validateState(Object.values(physics.state), 'render-physics')) {
            self.postMessage({ 
                    type: 'render_data',
                    // physics.state is now directly from Wasm via physics.wasmInstance.get_state_js()
                payload: { 
                    state: physics.state, 
                    params: physics.params,
                    action: isValidNumber(latestAction) ? latestAction : 0,
                    totalSteps: (simulationMode === 'TRAINING' && !isObservingPolicyWhileTrainingPaused) ? totalSteps : undefined,
                    policyDiagnostics: lastPolicyDiagnostics, // Add policy diagnostics
                    isWarmup: simulationMode === 'TRAINING' && totalSteps < AGENT_WARMUP_STEPS // Use agent's warmup
                } 
            });
        }
    }

    // Pace the simulation loop
    if (isEffectivelyObserving) {
        // Aim for roughly 60 physics steps per second to match dt = 1/60s for real-time playback
        setTimeout(runSimulation, 1000 / 60); 
    } else {
        // For training or other modes, run as fast as possible to maximize throughput
        setTimeout(runSimulation, 0);
    }
}
// `runSimulation` is an alias for `runSimulationLoop`, used in the setTimeout calls within `runSimulationLoop`.
const runSimulation = runSimulationLoop; 

// Worker message handler
self.onmessage = function(e) {
    const { type, payload } = e.data;

    switch(type) {
        case 'start_training':
            if (!agent) init();
            simulationMode = 'TRAINING';
            isPaused = false;
            isObservingPolicyWhileTrainingPaused = false;
            allowRender = payload && typeof payload.renderEnabled !== 'undefined' ? payload.renderEnabled : false;
            lastSpsCheckTime = performance.now(); // Reset SPS counters on start
            stepsSinceLastSpsCheck = 0;
            allowRenderBeforeTrainingPause = allowRender; 

            // Send initial agent config with training_started message
            const initialAgentConfig = agent ? {
                alpha: agent.alpha,
                actorLr: agent.actorLr,
                criticLr: agent.criticLr,
                batchSize: agent.batchSize,
                tau: agent.tau,
                gamma: agent.gamma,
                bufferSize: agent.bufferSize,
                warmupSteps: AGENT_WARMUP_STEPS, // Use the agent's actual warmup steps
                trainFrequency: TRAIN_FREQUENCY,
                stateSize: agent.stateSize,
                actionSize: agent.actionSize,
                alphaLr: agent.alphaLr,
                targetEntropy: agent.targetEntropy
            } : null;

            self.postMessage({ type: 'training_started', payload: { status: 'Training Active', agentConfig: initialAgentConfig } });
            runSimulationLoop();
            break;
        case 'stop_training_and_observe':
            if (agent && agent.isReady) {
                simulationMode = 'OBSERVING';
                isPaused = false;
                isObservingPolicyWhileTrainingPaused = false;
                state = physics.reset(); // Reset physics state for observation
                latestAction = 0;        // Reset last action as state changed
                allowRender = true;
                lastSpsCheckTime = performance.now(); // Reset SPS counters
                stepsSinceLastSpsCheck = 0;
                self.postMessage({ type: 'sps_update', payload: { sps: (60).toFixed(0) } }); // Initial SPS for observe
                self.postMessage({ type: 'observation_started', payload: { status: 'Observation Mode Active' } });
                // Ensure loop runs if it was IDLE or PAUSED from a non-running state
                if (previousSimulationMode === 'IDLE' || (isPaused && !isObservingPolicyWhileTrainingPaused)) runSimulationLoop();
            }
            break;
        case 'pause_simulation':
            if (!isPaused) { // Only pause if not already paused
                previousSimulationMode = simulationMode;
                isPaused = true;
                if (simulationMode === 'TRAINING') {
                    isObservingPolicyWhileTrainingPaused = true;
                    allowRenderBeforeTrainingPause = allowRender; // Save current render state
                    allowRender = true; // Force render for observing policy
                    self.postMessage({ type: 'simulation_paused', payload: { status: 'Paused (Observing Policy)', episode, totalSteps, originalMode: 'TRAINING' } });
                    // Loop continues for observing policy
                } else if (simulationMode === 'OBSERVING') {
                    isObservingPolicyWhileTrainingPaused = false; // Not training, so just a hard pause
                    self.postMessage({ type: 'simulation_paused', payload: { status: 'Observation Paused', episode, totalSteps, originalMode: 'OBSERVING' } });
                    // Loop will stop due to isPaused && !isObservingPolicyWhileTrainingPaused
                }
            }
            break;
        case 'resume_simulation':
            if (isPaused && agent && agent.isReady) {
                isPaused = false;
                simulationMode = previousSimulationMode;
                if (isObservingPolicyWhileTrainingPaused) { // If we were observing policy during a training pause
                    allowRender = allowRenderBeforeTrainingPause; // Restore original render setting for training
                }
                isObservingPolicyWhileTrainingPaused = false;
                self.postMessage({ type: 'simulation_resumed', payload: { status: 'Simulation Resumed', mode: simulationMode, renderEnabled: allowRender } });
                runSimulationLoop();
            }
            break;
        case 'reset_agent':
            simulationMode = 'IDLE'; // isPaused and isObservingPolicyWhileTrainingPaused will be reset by init()
            init();
            self.postMessage({ type: 'sps_update', payload: { sps: 0 } }); // Clear SPS on UI
            self.postMessage({ type: 'reset_complete', payload: { status: 'Agent Reset' } });
            break;
        case 'set_speed':
            userSetStepsPerFrame = payload.speed;
            // currentSimStepsPerFrame will be updated at the start of runSimulationLoop
            break;
        case 'set_render_enabled':
            allowRender = !!payload.enabled;
            break;
        case 'reset_pendulum_physics_state_only':
            if (physics) {
                physics.reset();
                state = physics.getStateArray();
                latestAction = 0;
                // Send an immediate render update with the new state
                if (allowRender) {
                    self.postMessage({ type: 'render_data', payload: { state: physics.state, params: physics.params, action: latestAction, totalSteps: simulationMode === 'TRAINING' ? totalSteps : undefined, isWarmup: simulationMode === 'TRAINING' && totalSteps < AGENT_WARMUP_STEPS } });
                }
            }
            break;
        // Cases 'start', 'stop', 'pause', 'resume', 'reset', 'set_render' are replaced or adapted.
        // Keep 'save_state', 'load_state', 'get_memory_info', 'force_gc'
        case 'save_state':
            if (agent && agent.isReady) {
                try {
                    // Create state snapshot
                    const agentState = {
                        episode,
                        totalSteps,
                        totalReward,
                        episodeRewards: episodeRewards.slice(-100), // Keep last 100 episodes
                        bestReward,
                        stateNormalization: {
                            mean: agent.stateRunningMean,
                            var: agent.stateRunningVar,
                            count: agent.stateCount
                        },
                        physics: {
                            state: physics.state,
                            currentStep: physics.currentStep
                        },
                        simulationMode: simulationMode, // Save current mode
                        isPaused: isPaused, isObservingPolicyWhileTrainingPaused: isObservingPolicyWhileTrainingPaused,
                        timestamp: Date.now()
                    };
                    
                    self.postMessage({ 
                        type: 'state_saved', 
                        payload: { 
                            status: 'State Saved Successfully',
                            agentState,
                            serializedSize: JSON.stringify(agentState).length
                        } 
                    });
                } catch (error) {
                    self.postMessage({ 
                        type: 'state_save_error', 
                        payload: { status: 'Failed to Save State', error: error.message } 
                    });
                }
            }
            break;
        case 'load_state':
            if (payload.agentState) {
                try {
                    // Restore state
                    episode = payload.agentState.episode || 0;
                    totalSteps = payload.agentState.totalSteps || 0;
                    totalReward = payload.agentState.totalReward || 0;
                    episodeRewards = payload.agentState.episodeRewards || [];
                    bestReward = payload.agentState.bestReward || -Infinity;
                    
                    // Restore normalization stats
                    if (agent && payload.agentState.stateNormalization) {
                        agent.stateRunningMean = payload.agentState.stateNormalization.mean;
                        agent.stateRunningVar = payload.agentState.stateNormalization.var;
                        agent.stateCount = payload.agentState.stateNormalization.count;
                    }
                    
                    // Restore physics state
                    if (payload.agentState.physics) {
                        // physics.state = payload.agentState.physics.state; // Wasm manages its own state
                        // TODO: Need a method in Wasm to set its state if loading. For now, it resets.
                        physics.currentStep = payload.agentState.physics.currentStep;
                    }
                    // Restore simulation mode if relevant
                    // simulationMode = payload.agentState.simulationMode || 'IDLE';
                    // isPaused = payload.agentState.isPaused || false;
                    // isObservingPolicyWhileTrainingPaused = payload.agentState.isObservingPolicyWhileTrainingPaused || false;
                    // For simplicity, loading state might imply going to IDLE or OBSERVING.
                    // Current logic will reset to IDLE on next action or keep current mode.
                    
                    state = physics.getStateArray();
                    
                    self.postMessage({ 
                        type: 'state_loaded', 
                        payload: { 
                            status: 'State Loaded Successfully',
                            episode,
                            totalSteps,
                            bestReward
                        } 
                    });
                } catch (error) {
                    self.postMessage({ 
                        type: 'state_load_error', 
                        payload: { status: 'Failed to Load State', error: error.message } 
                    });
                }
            }
            break;
        case 'get_memory_info':
            if (typeof tf !== 'undefined' && tf.memory) {
                const memInfo = tf.memory();
                self.postMessage({ 
                    type: 'memory_info', 
                    payload: { 
                        tensors: memInfo.numTensors,
                        bytes: memInfo.numBytes,
                        bufferSize: agent ? agent.replayBuffer.length : 0
                    } 
                });
            }
            break;
        case 'force_gc':
            // Force garbage collection
            if (typeof tf !== 'undefined' && tf.disposeVariables) {
                // Clean up any orphaned tensors
                const before = tf.memory().numTensors;
                tf.tidy(() => {}); // This helps clean up
                const after = tf.memory().numTensors;
                
                self.postMessage({ 
                    type: 'gc_complete', 
                    payload: { 
                        status: 'Garbage Collection Complete',
                        tensorsBefore: before,
                        tensorsAfter: after,
                        cleaned: before - after
                    } 
                });
            }
            break;
    }
};
