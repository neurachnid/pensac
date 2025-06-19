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
        // Parameters aligned with the reference paper
        const p = {
            cart_m: 0.350,
            m1: 0.133,
            m2: 0.025,
            l1_m: 0.5,
            l2_m: 0.5,
            g: 9.81
        };
        // For --target no-modules, WasmPendulumPhysics becomes a global constructor.
        // self.WasmPendulumPhysics or just WasmPendulumPhysics should work.
        this.wasmInstance = new wasm_bindgen.WasmPendulumPhysics(p.cart_m, p.m1, p.m2, p.l1_m, p.l2_m, p.g);
        this.params = this.wasmInstance.get_params_js(); // Get params from Wasm
        this.state = this.wasmInstance.get_state_js();   // Get initial state from Wasm
        // The paper applies forces in the range [-15, 15] N
        this.actionMax = 15.0;
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
        this.lastTerminationReason = 'N/A';
        return this.getStateArray();
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
        const dt = 0.02; // 50 Hz physics timestep
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
            } else if (Math.abs(this.state.cart_x_m) > 2.4) {
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
        const { a1, a2, cart_x_m } = this.state;

        // Check for valid numbers
        const varsToCheck = [a1, a2, cart_x_m, action];
        if (varsToCheck.some(v => !isFinite(v) || isNaN(v))) {
            console.warn('NaN/Inf in reward inputs', varsToCheck);
            return -10;
        }

        // Convert internal angles (0 = down) to paper convention (0 = up)
        const theta1 = a1 - Math.PI;
        const theta2 = a2 - Math.PI;

        // Reward weights from the reference paper
        const w0 = 0.1;
        const w1 = 5.0;
        const w2 = 5.0;
        const w3 = 1.0;
        const w4 = 0.05;
        const Vp = 100.0;

        const penalty = w1 * theta1 * theta1 +
                        w2 * theta2 * theta2 +
                        w3 * cart_x_m * cart_x_m +
                        w4 * (action * action);

        const F = Math.abs(cart_x_m) > 2.4 ? 1.0 : 0.0;

        const r = -w0 * penalty - Vp * F;

        this.lastRewardComponents = { penalty: -w0 * penalty, outOfBounds: -Vp * F };
        this.effectiveRewardWeights = { w0, w1, w2, w3, w4, Vp };

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

// --- DDPG Agent Implementation ---
class DDPGAgent {
    constructor() {
        this.stateSize = 8;
        this.actionSize = 1;
        this.actionBounds = 1.0; // actions in [-1,1]

        this.actorLr = 1e-4;
        this.criticLr = 1e-3;
        this.gamma = 0.99;
        this.tau = 0.005;
        this.batchSize = 256;
        this.bufferSize = 1000000; // Replay buffer of 1M transitions
        this.warmupSteps = AGENT_WARMUP_STEPS; // keep in sync with constant

        this.replayBuffer = [];
        this.isReady = false;

        this.stateRunningMean = new Array(this.stateSize).fill(0);
        this.stateRunningVar = new Array(this.stateSize).fill(1);
        this.stateCount = 0;

        this.init();
    }

    async init() {
        tf.serialization.registerClass(ScaleLayer);

        this.actor = this.buildActor();
        this.critic = this.buildCritic();
        this.targetActor = this.buildActor();
        this.targetCritic = this.buildCritic();

        this.actorOptimizer = tf.train.adam(this.actorLr);
        this.criticOptimizer = tf.train.adam(this.criticLr);

        this.updateTargetNetworks(1.0);
        this.isReady = true;
    }

    buildActor() {
        const input = tf.input({shape: [this.stateSize]});
        let x = tf.layers.dense({units: 400, activation: 'relu'}).apply(input);
        x = tf.layers.dense({units: 300, activation: 'relu'}).apply(x);
        const out = tf.layers.dense({units: this.actionSize, activation: 'tanh'}).apply(x);
        const scaled = new ScaleLayer({scaleFactor: this.actionBounds}).apply(out);
        return tf.model({inputs: input, outputs: scaled});
    }

    buildCritic() {
        const stateInput = tf.input({shape: [this.stateSize]});
        const actionInput = tf.input({shape: [this.actionSize]});
        let s = tf.layers.dense({units: 400, activation: 'relu'}).apply(stateInput);
        const concat = tf.layers.concatenate().apply([s, actionInput]);
        let x = tf.layers.dense({units: 300, activation: 'relu'}).apply(concat);
        const out = tf.layers.dense({units: 1}).apply(x);
        return tf.model({inputs: [stateInput, actionInput], outputs: out});
    }

    normalizeState(state) {
        const normalized = [];
        this.stateCount++;
        const count = Math.min(this.stateCount, 10000);
        for (let i = 0; i < state.length; i++) {
            const val = Math.max(-1000, Math.min(1000, state[i]));
            const delta = val - this.stateRunningMean[i];
            this.stateRunningMean[i] += delta / count;
            const delta2 = val - this.stateRunningMean[i];
            this.stateRunningVar[i] = (this.stateRunningVar[i] * (count - 1) + delta * delta2) / count;
            const std = Math.sqrt(Math.max(this.stateRunningVar[i], 1e-6));
            normalized.push((val - this.stateRunningMean[i]) / std);
        }
        return normalized;
    }

    chooseAction(state, deterministic = false) {
        return tf.tidy(() => {
            const normState = this.normalizeState(state);
            const s = tf.tensor([normState]);
            let action = this.actor.predict(s).flatten();
            let val = action.dataSync()[0];
            if (!deterministic) {
                val += (Math.random() * 2 - 1) * 0.1;
            }
            return Math.max(-1, Math.min(1, val));
        });
    }

    remember(state, action, reward, nextState, done) {
        if (this.replayBuffer.length >= this.bufferSize) {
            this.replayBuffer.shift();
        }
        this.replayBuffer.push({state, action, reward, nextState, done});
    }

    sampleBatch() {
        const batch = {states: [], actions: [], rewards: [], nextStates: [], dones: []};
        for (let i = 0; i < this.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.replayBuffer.length);
            const e = this.replayBuffer[idx];
            batch.states.push(e.state);
            batch.actions.push([e.action]);
            batch.rewards.push([e.reward]);
            batch.nextStates.push(e.nextState);
            batch.dones.push([e.done ? 1 : 0]);
        }
        return batch;
    }

    train() {
        if (this.replayBuffer.length < this.batchSize) return {};

        const b = this.sampleBatch();
        const states = tf.tensor(b.states);
        const actions = tf.tensor(b.actions);
        const rewards = tf.tensor(b.rewards);
        const nextStates = tf.tensor(b.nextStates);
        const dones = tf.tensor(b.dones);

        const criticLossTensor = this.criticOptimizer.minimize(() => {
            const nextActions = this.targetActor.predict(nextStates);
            const qNext = this.targetCritic.predict([nextStates, nextActions]).reshape([this.batchSize]);
            const y = rewards.reshape([this.batchSize]).add(
                dones.reshape([this.batchSize]).mul(-1).add(1).mul(this.gamma).mul(qNext)
            );
            const q = this.critic.predict([states, actions]).reshape([this.batchSize]);
            return tf.losses.meanSquaredError(y, q);
        }, true);

        const actorLossTensor = this.actorOptimizer.minimize(() => {
            const act = this.actor.predict(states);
            const qVal = this.critic.predict([states, act]).reshape([this.batchSize]);
            return tf.neg(tf.mean(qVal));
        }, true);

        this.updateTargetNetworks(this.tau);

        const criticLoss = criticLossTensor ? criticLossTensor.dataSync()[0] : 0;
        const actorLoss = actorLossTensor ? actorLossTensor.dataSync()[0] : 0;

        // Dispose tensors to prevent memory leaks
        states.dispose();
        actions.dispose();
        rewards.dispose();
        nextStates.dispose();
        dones.dispose();
        if (criticLossTensor) criticLossTensor.dispose();
        if (actorLossTensor) actorLossTensor.dispose();

        return {criticLoss, actorLoss};
    }

    updateTargetNetworks(tau) {
        const updateTarget = (target, source) => {
            const tw = target.getWeights();
            const sw = source.getWeights();
            const newW = sw.map((w, i) => tf.tidy(() => w.mul(tau).add(tw[i].mul(1 - tau))));
            target.setWeights(newW);
            // Dispose only the temporary tensors created in newW.
            // The arrays from getWeights() (tw, sw) contain the model's
            // actual weight variables and should **not** be disposed here.
            tf.dispose(newW);
        };
        updateTarget(this.targetActor, this.actor);
        updateTarget(this.targetCritic, this.critic);
    }

    getPolicyDiagnostics() {
        return {};
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
const MAX_EPISODE_STEPS = 1000; // Match reference paper
const AGENT_WARMUP_STEPS = 1000; // Should match agent.warmupSteps
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
    agent = new DDPGAgent();
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
    console.log('Initializing DDPG agent for double pendulum...');
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
                episodeSteps: physics.currentStep,
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
                actorLr: agent.actorLr,
                criticLr: agent.criticLr,
                batchSize: agent.batchSize,
                tau: agent.tau,
                gamma: agent.gamma,
                bufferSize: agent.bufferSize,
                warmupSteps: AGENT_WARMUP_STEPS,
                trainFrequency: TRAIN_FREQUENCY,
                stateSize: agent.stateSize,
                actionSize: agent.actionSize,
                stateNormalizationMean: agent.stateRunningMean.map(v => parseFloat(v.toFixed(4))),
                stateNormalizationVar: agent.stateRunningVar.map(v => parseFloat(v.toFixed(4)))
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
        // For observing mode, SPS is roughly 50 due to setTimeout aiming for ~20ms per frame (and 1 step per frame)
        if (now - lastSpsCheckTime >= 990) {
            self.postMessage({ type: 'sps_update', payload: { sps: (1 * 50).toFixed(0) } }); // Approx 50 SPS
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
        // Aim for roughly 50 physics steps per second to match dt = 0.02s for real-time playback
        setTimeout(runSimulation, 1000 / 50);
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
                actorLr: agent.actorLr,
                criticLr: agent.criticLr,
                batchSize: agent.batchSize,
                tau: agent.tau,
                gamma: agent.gamma,
                bufferSize: agent.bufferSize,
                warmupSteps: AGENT_WARMUP_STEPS,
                trainFrequency: TRAIN_FREQUENCY,
                stateSize: agent.stateSize,
                actionSize: agent.actionSize
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
                self.postMessage({ type: 'sps_update', payload: { sps: (50).toFixed(0) } }); // Initial SPS for observe
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
