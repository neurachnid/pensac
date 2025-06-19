// --- Main UI Thread Script ---

// --- Main Simulation and Environment Class ---
class PendulumRenderer {
    constructor() {
        this.pendulumCanvas = document.getElementById('pendulumCanvas');
        this.p_ctx = this.pendulumCanvas.getContext('2d');
        this.traceCanvas = document.getElementById('traceCanvas');
        this.t_ctx = this.traceCanvas.getContext('2d');
        // Physical parameters from the referenced DDPG paper
        this.params = {
            cart_m: 0.350,
            m1: 0.133,
            m2: 0.025,
            l1_m: 0.5,
            l2_m: 0.5,
            g: 9.81
        };
        this.state = null; // This will be updated from worker
        this.camera_x_m = 0;

        this.pixelsPerMeter = 100; // Initial default, will be refined
        this.py = 300; // Vertical pivot point on canvas
        this.minPixelsPerMeter = 10; // Maximum zoom out
        this.maxPixelsPerMeter = 150; // Maximum zoom in (will be calculated dynamically)
        
        this.resizeAndResetCanvas();
        window.addEventListener('resize', () => this.resizeAndResetCanvas());
    }
    
    updateData(state, params) {
        // It's possible params could change if we implement dynamic param updates later
        // If params change, maxPixelsPerMeter might need re-evaluation or resizeAndResetCanvas called.
        // For now, assuming params are static after init for simplicity of zoom logic.
        this.state = state;
        this.params = params;
    }

    calculatePositions() { if(!this.state) return {}; const {l1_m, l2_m} = this.params; const {a1, a2, cart_x_m} = this.state; if(isNaN(cart_x_m)) return {}; const l1_px = l1_m * this.pixelsPerMeter; const l2_px = l2_m * this.pixelsPerMeter; const px = this.pendulumCanvas.width / 2 + (cart_x_m - this.camera_x_m) * this.pixelsPerMeter; const x1 = px + l1_px * Math.sin(a1); const y1 = this.py + l1_px * Math.cos(a1); const x2 = x1 + l2_px * Math.sin(a2); const y2 = y1 + l2_px * Math.cos(a2); return {x1, y1, x2, y2, px}; }
    render() { this.updateCamera(1/50); this.draw(); }
    resizeAndResetCanvas() { const container = document.getElementById('simulation-container'); const containerWidth = container.clientWidth; const totalLengthMeters = this.params.l1_m + this.params.l2_m; const fullDrawableHeight = 600; this.pixelsPerMeter = (totalLengthMeters > 0) ? (fullDrawableHeight / (totalLengthMeters * 2.2)) : 100; this.py = fullDrawableHeight / 2; container.style.height = (fullDrawableHeight) + 'px'; this.pendulumCanvas.height = fullDrawableHeight; this.pendulumCanvas.width = containerWidth; this.traceCanvas.height = fullDrawableHeight; this.traceCanvas.width = containerWidth; this.drawGrid(); this.draw(); }
    draw() { 
        this.p_ctx.clearRect(0, 0, this.pendulumCanvas.width, this.pendulumCanvas.height); 
        this.drawGrid(); 
        if(!this.state) return; 
        const {x1, y1, x2, y2, px} = this.calculatePositions(); 
        if (px === undefined || isNaN(px)) return; // Added NaN check for px
        const cartWidth = Math.max(20, 0.08 * this.pixelsPerMeter); // Scale cart size with zoom, min 20px
        const cartHeight = Math.max(10, 0.04 * this.pixelsPerMeter); // Scale cart size with zoom, min 10px
        this.p_ctx.fillStyle = '#4b5563'; 
        this.p_ctx.fillRect(px - cartWidth / 2, this.py - cartHeight / 2, cartWidth, cartHeight); 
        this.p_ctx.beginPath(); this.p_ctx.arc(px, this.py, Math.max(3, 0.015 * this.pixelsPerMeter), 0, 2 * Math.PI); this.p_ctx.fillStyle = '#9ca3af'; this.p_ctx.fill(); 
        this.p_ctx.beginPath(); this.p_ctx.moveTo(px, this.py); this.p_ctx.lineTo(x1, y1); this.p_ctx.lineTo(x2, y2); this.p_ctx.strokeStyle = '#6b7280'; this.p_ctx.lineWidth = Math.max(1, 0.005 * this.pixelsPerMeter); this.p_ctx.stroke(); 
        const r_base = 0.01 * this.pixelsPerMeter; // Base radius for masses, scaled
        const r1 = Math.max(2, Math.pow(this.params.m1, 1/3) * r_base + r_base); const r2 = Math.max(2, Math.pow(this.params.m2, 1/3) * r_base + r_base); 
        // Adjusted bob radius calculation for better visual appearance
        const visualMassScaleFactor = 0.075; // Increase for larger bobs (e.g., 0.075 means a 1kg mass would have a base visual radius of 7.5cm)
        const minBobRadiusPx = 4; // Minimum pixel radius for bobs
        const visual_r1 = Math.max(minBobRadiusPx, Math.pow(this.params.m1, 1/3) * visualMassScaleFactor * this.pixelsPerMeter);
        const visual_r2 = Math.max(minBobRadiusPx, Math.pow(this.params.m2, 1/3) * visualMassScaleFactor * this.pixelsPerMeter);
        this.p_ctx.beginPath(); this.p_ctx.arc(x1, y1, visual_r1, 0, 2 * Math.PI); const isTraining = document.getElementById('trainButton').textContent === 'Stop Training'; this.p_ctx.fillStyle = isTraining ? '#3b82f6' : '#60a5fa'; this.p_ctx.fill(); 
        this.p_ctx.beginPath(); this.p_ctx.arc(x2, y2, visual_r2, 0, 2 * Math.PI); this.p_ctx.fillStyle = isTraining ? '#ef4444' : '#f87171'; this.p_ctx.fill(); 
    }
    drawGrid() { this.t_ctx.clearRect(0, 0, this.traceCanvas.width, this.traceCanvas.height); this.t_ctx.fillStyle = '#1f2937'; this.t_ctx.fillRect(0, 0, this.traceCanvas.width, this.traceCanvas.height); const meterInPixels = 1 * this.pixelsPerMeter; const lineColor = 'rgba(75, 85, 99, 0.5)'; this.t_ctx.strokeStyle = lineColor; this.t_ctx.font = '12px Inter'; this.t_ctx.fillStyle = lineColor; const start_x_m = Math.floor(this.camera_x_m - (this.traceCanvas.width / 2 / this.pixelsPerMeter)); const end_x_m = Math.ceil(this.camera_x_m + (this.traceCanvas.width / 2 / this.pixelsPerMeter)); for(let i = start_x_m; i <= end_x_m; i++) { const x = this.traceCanvas.width / 2 + (i - this.camera_x_m) * meterInPixels; this.t_ctx.beginPath(); this.t_ctx.lineWidth = (i % 5 === 0) ? 1.5 : 0.5; this.t_ctx.moveTo(x, 0); this.t_ctx.lineTo(x, this.traceCanvas.height); this.t_ctx.stroke(); if (i % 5 === 0 && i !== 0) this.t_ctx.fillText(`${i}m`, x + 5, 20); } }
    updateCamera(dt) { 
        if(!this.state || !this.pendulumCanvas || this.pendulumCanvas.width === 0 || this.pendulumCanvas.height === 0) return; 

        // Lock camera x-position directly to the cart's x-position
        this.camera_x_m = this.state.cart_x_m;

        // pixelsPerMeter is now static, set by resizeAndResetCanvas.
        // No dynamic zoom logic needed here.
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const worker = new Worker('worker.js');
    
    const renderer = new PendulumRenderer();
    // Main control buttons - pauseResumeButton will be repurposed
    const trainButton = document.getElementById('trainButton');
    const pauseResumeButton = document.getElementById('pauseResumeButton'); // Updated ID
    const resetButton = document.getElementById('resetButton');
    const renderButton = document.getElementById('renderButton');
    const saveStateButton = document.getElementById('saveStateButton');
    const loadStateButton = document.getElementById('loadStateButton');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    const copyDebugInfoButton = document.getElementById('copyDebugInfoButton'); // Assuming this ID exists in your HTML
    const spsDisplay = document.getElementById('sps-display');
    const rewardChartCtx = document.getElementById('rewardChart').getContext('2d');
    // Assume lossChart canvas exists in HTML
    const lossChartCtx = document.getElementById('lossChart') ? document.getElementById('lossChart').getContext('2d') : null;

    let workerAppMode = 'IDLE'; // IDLE, TRAINING, OBSERVING, PAUSED_TRAINING_SHOWING_POLICY, PAUSED_OBSERVING_STATIC
    let animationFrameId;

    let isRendering = true;
    
    let rewardChartInstance;
    let lossChartInstance; // For Actor/Critic losses
    let currentChunkRewards = [];
    let fullRewardHistory = [];
    const CHART_UPDATE_INTERVAL = 25; // Less frequent updates
    const MAX_CHART_POINTS = 500; // Fewer data points
    let chartNeedsUpdate = false;
    let bestRewardEver = -Infinity;
    let latestDebugSnapshot = {}; // To store all data for the debug panel
    let actorLossHistory = [];
    let criticLossHistory = [];
    // let avgLogProbHistory = []; // Could add another chart for this

    worker.onmessage = (e) => {
        const { type, payload } = e.data;
        switch(type) {
            case 'render_data':
                // Worker pushed new data. Update the renderer.
                if (isRendering) renderer.updateData(payload.state, payload.params);
                // Update parts of the debug snapshot that come with render_data
                latestDebugSnapshot.lastAction = payload.action;
                latestDebugSnapshot.isWarmup = payload.isWarmup;
                if (payload.totalSteps !== undefined) latestDebugSnapshot.totalSteps = payload.totalSteps;
                // Avoid calling updateDebugInfoPanel here for performance, rely on episode_done or other events
                // Or, if frequent updates are needed, ensure updateDebugInfoPanel is efficient.
                // For now, let's update it less frequently.
                
                // Update additional training info
                latestDebugSnapshot.physicsParams = payload.params; // Store physics params
                if (payload.totalSteps !== undefined) {
                    // document.getElementById('total-steps').textContent = payload.totalSteps.toLocaleString(); // Updated by episode_done
                    
                    // const trainingStatus = payload.isWarmup ?  // This is better handled by updateDebugInfoPanel
                    //     `Warming up... (${payload.totalSteps}/${latestDebugSnapshot.agentConfig?.warmupSteps || AGENT_WARMUP_STEPS_FALLBACK})` :
                    //     'Training';
                    // document.getElementById('training-status').textContent = trainingStatus;
                }
                
                // Update last action display
                if (payload.action !== undefined) {
                    document.getElementById('lastAction').textContent = payload.action.toFixed(3);
                }
                break;
                
            case 'episode_done':
                const {
                    totalReward, episode, bestReward, avgReward,
                    totalSteps, bufferSize, mode, episodeSteps,
                    // Removed trainingLosses, agentConfig, physicsRewardConfig from payload destructuring
                    currentSpeed
                } = payload; // payload is e.data.payload

                // Store all data for the debug panel
                latestDebugSnapshot = { ...latestDebugSnapshot, ...payload };
                // Correctly access top-level properties from e.data for the snapshot
                // These are now directly on e.data, not e.data.payload
                if (e.data.trainingLosses) latestDebugSnapshot.trainingLosses = e.data.trainingLosses;
                // Add the new ones from e.data (top level) that are now also sent with episode_done
                if (e.data.lastStepRewardComponents) latestDebugSnapshot.lastStepRewardComponents = e.data.lastStepRewardComponents;
                if (e.data.lastAction !== undefined) latestDebugSnapshot.lastAction = e.data.lastAction;
                if (e.data.terminationReason) latestDebugSnapshot.terminationReason = e.data.terminationReason;
                if (e.data.agentConfig) latestDebugSnapshot.agentConfig = e.data.agentConfig; 
                if (e.data.physicsRewardConfig) latestDebugSnapshot.physicsRewardConfig = e.data.physicsRewardConfig;
                if (episodeSteps !== undefined) latestDebugSnapshot.episodeSteps = episodeSteps;

                // Update stats
                document.getElementById('episode-counter').textContent = episode;
                document.getElementById('total-reward').textContent = totalReward.toFixed(2);
                document.getElementById('best-reward').textContent = bestReward.toFixed(2);
                document.getElementById('avg-reward').textContent = avgReward.toFixed(2);
                
                // Update agent-specific info (only if in training mode for these stats)
                if (totalSteps !== undefined) {
                    document.getElementById('total-steps').textContent = totalSteps.toLocaleString();
                }
                
                if (bufferSize !== undefined) {
                    document.getElementById('buffer-size').textContent = bufferSize.toLocaleString();
                    const bufferProgress = `${bufferSize.toLocaleString()}/1M`;
                    document.getElementById('buffer-progress').textContent = bufferProgress;
                }
                
                // Update training status
                if (mode === 'TRAINING' || mode === 'TRAINING_PAUSED_OBSERVING') {
                    const trainingStatus = totalSteps < 1000 ? 
                        `Warming up... (${totalSteps}/1000)` : 
                        'Training';
                    document.getElementById('training-status').textContent = trainingStatus;
                }

                // Track best reward ever for visual feedback
                if ((mode === 'TRAINING' || mode === 'TRAINING_PAUSED_OBSERVING') && bestReward > bestRewardEver) {
                    bestRewardEver = bestReward;
                    document.getElementById('best-reward').classList.add('animate-pulse');
                    setTimeout(() => {
                        document.getElementById('best-reward').classList.remove('animate-pulse');
                    }, 2000);
                }

                // Update chart data less frequently
                fullRewardHistory.push({ episode, reward: totalReward, avgReward, bestReward });
                if (fullRewardHistory.length % CHART_UPDATE_INTERVAL === 0) {
                    chartNeedsUpdate = true;
                }

                // Correctly use e.data.trainingLosses for chart history
                if (e.data.trainingLosses && (mode === 'TRAINING' || mode === 'TRAINING_PAUSED_OBSERVING')) {
                    if (typeof e.data.trainingLosses.actorLoss === 'number') actorLossHistory.push({ episode, loss: e.data.trainingLosses.actorLoss });
                    if (typeof e.data.trainingLosses.criticLoss === 'number') criticLossHistory.push({ episode, loss: e.data.trainingLosses.criticLoss });
                    // if (typeof trainingLosses.avgLogProb === 'number') avgLogProbHistory.push({ episode, value: trainingLosses.avgLogProb });
                    if (actorLossHistory.length > 0 || criticLossHistory.length > 0) chartNeedsUpdate = true; // Ensure chart updates if new loss data
                    // Chart update will be triggered by chartNeedsUpdate for rewards, can piggyback or use separate flag
                }

                updateDebugInfoPanel(); // Update the debug panel on episode end
                break;
                
            case 'training_started':
                workerAppMode = 'TRAINING';
                document.getElementById('currentMode').textContent = 'Training Active';
                document.getElementById('currentMode').className = 'text-green-400 font-bold';
                document.getElementById('agentState').textContent = 'Learning';
                trainButton.textContent = 'Stop Training';
                trainButton.textContent = 'Pause Training'; // User request
                pauseResumeButton.disabled = false; // This is now resetPendulumButton
                speedSlider.disabled = false;
                trainButton.disabled = false;
                saveStateButton.disabled = false;

                if (payload.agentConfig) {
                    latestDebugSnapshot.agentConfig = payload.agentConfig;
                }
                latestDebugSnapshot.mode = 'TRAINING';
                // Initialize totalSteps and isWarmup for the debug panel if not already present
                if (latestDebugSnapshot.agentConfig && latestDebugSnapshot.totalSteps === undefined) {
                    latestDebugSnapshot.totalSteps = 0; // Assume 0 at the very start
                }
                if (latestDebugSnapshot.agentConfig && latestDebugSnapshot.totalSteps !== undefined) {
                    latestDebugSnapshot.isWarmup = latestDebugSnapshot.totalSteps < latestDebugSnapshot.agentConfig.warmupSteps;
                }
                updateDebugInfoPanel();
                break;
                
            case 'observation_started':
                workerAppMode = 'OBSERVING';
                document.getElementById('currentMode').textContent = 'Observing';
                document.getElementById('currentMode').className = 'text-blue-400 font-bold';
                document.getElementById('agentState').textContent = 'Playback';
                trainButton.textContent = 'Start Training';
                pauseResumeButton.disabled = false; // Enable Reset Pendulum Position button in observe mode
                speedSlider.disabled = true; // Disable slider in observation mode
                trainButton.disabled = false;
                saveStateButton.disabled = false;
                // Ensure visuals are on for observation
                isRendering = true;
                renderButton.textContent = 'Pause Visuals';
                latestDebugSnapshot.mode = 'OBSERVING';
                updateDebugInfoPanel();
                break;
                
            case 'simulation_paused':
                // payload.originalMode is 'TRAINING' or 'OBSERVING'
                // payload.status is 'Paused (Observing Policy)' or 'Observation Paused'
                document.getElementById('currentMode').textContent = payload.status; // e.g., "Paused (Observing Policy)"
                document.getElementById('currentMode').className = 'text-yellow-400 font-bold';
                document.getElementById('agentState').textContent = `Paused (Ep ${payload.episode || 0})`;

                if (payload.originalMode === 'TRAINING') {
                    workerAppMode = 'PAUSED_TRAINING_SHOWING_POLICY';
                    trainButton.textContent = 'Resume Training';
                    pauseResumeButton.disabled = false; // resetPendulumButton enabled if paused from training
                    isRendering = true; // Force rendering to see the policy
                    renderButton.textContent = 'Pause Visuals';
                    speedSlider.disabled = true; // Speed is fixed to observation speed
                } else { // OBSERVING
                    workerAppMode = 'PAUSED_OBSERVING_STATIC';
                    trainButton.textContent = 'Start Training'; // Or "Resume Observation" if we add that path
                    pauseResumeButton.disabled = false; // Enable Reset Pendulum Position button if paused from observing
                    speedSlider.disabled = true;
                }
                trainButton.disabled = false; // Allow resuming by keeping the button enabled
                showStatusMessage(payload.status, 'info');
                latestDebugSnapshot.mode = workerAppMode; 
                latestDebugSnapshot.pausedMode = payload.originalMode; // Store what mode it was before pausing
                updateDebugInfoPanel();
                break;
                
            case 'simulation_resumed':
                // payload.mode is 'TRAINING' or 'OBSERVING' (the mode it resumed TO)
                // payload.renderEnabled is the render state in the worker
                workerAppMode = payload.mode; 
                const resumedModeText = `${payload.mode.charAt(0).toUpperCase() + payload.mode.slice(1)} Active`;
                document.getElementById('currentMode').textContent = resumedModeText;
                document.getElementById('currentMode').className = payload.mode === 'TRAINING' ? 'text-green-400 font-bold' : 'text-blue-400 font-bold';
                document.getElementById('agentState').textContent = payload.mode === 'TRAINING' ? 'Learning' : 'Playback';
                
                if (payload.mode === 'TRAINING') {
                    trainButton.textContent = 'Pause Training';
                    pauseResumeButton.disabled = false; // resetPendulumButton enabled
                } else { // OBSERVING
                    trainButton.textContent = 'Start Training';
                    pauseResumeButton.disabled = false; // Enable Reset Pendulum Position if resuming to observation
                }
                
                speedSlider.disabled = (payload.mode === 'OBSERVING');
                trainButton.disabled = false;
                // Sync main thread's rendering state with worker's state upon resume
                isRendering = payload.renderEnabled;
                renderButton.textContent = isRendering ? 'Pause Visuals' : 'Resume Visuals';
                showStatusMessage(`${payload.mode.charAt(0).toUpperCase() + payload.mode.slice(1)} resumed`, 'success');
                latestDebugSnapshot.mode = payload.mode;
                updateDebugInfoPanel();
                break;
                
            case 'reset_complete':
                workerAppMode = 'IDLE';
                trainButton.textContent = 'Start Training';
                pauseResumeButton.disabled = true; // resetPendulumButton disabled
                speedSlider.disabled = false;
                trainButton.disabled = false;
                saveStateButton.disabled = true;
                resetUI();
                latestDebugSnapshot.mode = 'IDLE';
                updateDebugInfoPanel(); // Update debug panel on reset
                break;
                
            case 'state_saved':
                localStorage.setItem('doublePendulumState', JSON.stringify(payload.agentState));
                document.getElementById('loadStateButton').disabled = false;
                showStatusMessage(`State saved (${(payload.serializedSize / 1024).toFixed(1)}KB)`, 'success');
                break;
                
            case 'state_loaded':
                // Update UI with loaded state
                document.getElementById('episode-counter').textContent = payload.episode;
                document.getElementById('total-steps').textContent = payload.totalSteps.toLocaleString();
                document.getElementById('best-reward').textContent = payload.bestReward.toFixed(2);
                showStatusMessage('State loaded successfully', 'success');
                updateDebugInfoPanel(); // Refresh with potentially loaded data
                break;
                
            case 'state_save_error':
            case 'state_load_error':
                showStatusMessage(payload.status + ': ' + payload.error, 'error');
                break;
                
            case 'memory_info':
                const memoryDisplay = `${payload.tensors} tensors`;
                document.getElementById('memoryUsage').textContent = memoryDisplay;
                latestDebugSnapshot.memoryUsage = memoryDisplay; // Store for the debug panel
                // updateDebugInfoPanel(); // Optionally call if immediate update to panel is desired
                break;
            case 'sps_update':
                if (spsDisplay && payload.sps !== undefined) {
                    spsDisplay.textContent = `${payload.sps} SPS`;
                }
                latestDebugSnapshot.sps = payload.sps; // For debug panel
                break; // Add break to prevent fall-through
            case 'gc_complete':
                showStatusMessage(`Memory cleaned: ${payload.cleaned} tensors freed`, 'info');
                break;
        }
    };
    
    function showStatusMessage(message, type = 'info') {
        // Create temporary status message
        const statusDiv = document.createElement('div');
        statusDiv.className = `fixed top-4 right-4 p-4 rounded-lg z-50 transition-all duration-300 ${
            type === 'success' ? 'bg-green-600' : 
            type === 'error' ? 'bg-red-600' : 
            type === 'warning' ? 'bg-yellow-600' : 'bg-blue-600'
        } text-white`;
        statusDiv.textContent = message;
        document.body.appendChild(statusDiv);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            statusDiv.style.transform = 'translateX(100%)'; // Animate out
            setTimeout(() => statusDiv.remove(), 300); // Remove after animation
        }, 3000);
    }
    
    // Memory monitoring
    setInterval(() => {
        worker.postMessage({ type: 'get_memory_info' });
    }, 5000); // Check every 5 seconds

    function renderingLoop() {
        // Removed frame skip – render loop always runs at UI refresh
        if (chartNeedsUpdate) {
            updateRewardChartWithDownsampling();
            if (lossChartCtx) updateLossChartWithDownsampling();
            chartNeedsUpdate = false;
        }

        if (isRendering) {
            renderer.render();
        }
        
        animationFrameId = requestAnimationFrame(renderingLoop);
    }

    function updateDebugInfoPanel() {
        const panel = document.getElementById('debugInfoPanel'); // Assuming this ID exists in your HTML
        if (!panel) return;

        const data = latestDebugSnapshot;
        let content = `--- Simulation Status ---\n`; // Use the centrally stored snapshot
        content += `Mode: ${data.mode || 'IDLE'}`;
        if (data.mode === 'PAUSED' && data.pausedMode) {
            content += ` (was ${data.pausedMode})`;
        }
        content += `\n`;

        if ((data.mode === 'TRAINING' || (data.mode === 'PAUSED' && data.pausedMode === 'TRAINING')) && data.agentConfig) {
            let statusText = 'N/A';
            if (data.totalSteps !== undefined && data.agentConfig.warmupSteps !== undefined) {
                if (data.totalSteps < data.agentConfig.warmupSteps) {
                    statusText = `Warming up (${data.totalSteps || 0}/${data.agentConfig.warmupSteps})`;
                } else {
                    statusText = (data.mode === 'TRAINING') ? 'Training Active' : 'Paused (Observing Policy)';
                }
            }
            content += `Status: ${statusText}\n`;
        } else if (data.mode === 'OBSERVING' || (data.mode === 'PAUSED' && data.pausedMode === 'OBSERVING')) {
            const statusText = data.mode === 'OBSERVING' ? 'Observing' : 'Observation Paused';
            content += `Status: ${statusText}\n`;
        } else {
             content += `Status: N/A\n`;
        }
        if (data.terminationReason && data.terminationReason !== 'Running') content += `Last Termination: ${data.terminationReason}\n`;
        content += `Episode: ${data.episode || 0}\n`;
        if (data.episodeSteps !== undefined) content += `Episode Length: ${data.episodeSteps} steps\n`;
        content += `Total Steps: ${(data.totalSteps || 0).toLocaleString()}\n`;
        content += `Actual Sim Speed: ${data.sps || '-'} SPS\n`;
        content += `Sim Speed: x${data.currentSpeed || speedSlider.value || 1}\n`;

        content += `\n--- Rewards ---\n`;
        content += `Current Ep Reward: ${(data.totalReward || 0).toFixed(2)}\n`;
        content += `Best Ep Reward: ${(data.bestReward || -Infinity).toFixed(2)}\n`;
        content += `Avg Reward (last 10): ${(data.avgReward || 0).toFixed(2)}\n`;

        // Use latestDebugSnapshot.trainingLosses directly as it's updated from worker
        if (latestDebugSnapshot.trainingLosses) {
            content += `\n--- Last Training Batch ---\n`;
            content += `Actor Loss: ${latestDebugSnapshot.trainingLosses.actorLoss?.toFixed(4) || 'N/A'}\n`;
            content += `Critic Loss: ${latestDebugSnapshot.trainingLosses.criticLoss?.toFixed(4) || 'N/A'}\n`;
            if (latestDebugSnapshot.trainingLosses.actorGradNorm !== undefined) {
                content += `Actor Grad Norm: ${latestDebugSnapshot.trainingLosses.actorGradNorm.toFixed(4)}\n`;
            }
            if (latestDebugSnapshot.trainingLosses.criticGradNorm !== undefined) {
                content += `Critic Grad Norm: ${latestDebugSnapshot.trainingLosses.criticGradNorm.toFixed(4)}\n`;
            }
        }

        if (data.agentConfig) {
            content += `\n--- RL Agent Config ---\n`;
            if (data.agentConfig.alpha !== undefined) {
                content += `Alpha (Entropy): ${data.agentConfig.alpha}\n`;
            }
            if (data.agentConfig.actorLr !== undefined) content += `Actor LR: ${data.agentConfig.actorLr}\n`;
            if (data.agentConfig.criticLr !== undefined) content += `Critic LR: ${data.agentConfig.criticLr}\n`;
            if (data.agentConfig.tau !== undefined) content += `Tau (Target Update): ${data.agentConfig.tau}\n`;
            if (data.agentConfig.gamma !== undefined) content += `Gamma (Discount): ${data.agentConfig.gamma}\n`;
            if (data.agentConfig.batchSize !== undefined) content += `Batch Size: ${data.agentConfig.batchSize}\n`;
            if (data.agentConfig.bufferSize !== undefined) content += `Buffer: ${(data.bufferSize || 0).toLocaleString()} / ${data.agentConfig.bufferSize?.toLocaleString() || 'N/A'}\n`;
            if (data.agentConfig.warmupSteps !== undefined) content += `Warmup Steps: ${data.agentConfig.warmupSteps}\n`;
            if (data.agentConfig.trainFrequency !== undefined) content += `Train Freq (steps): ${data.agentConfig.trainFrequency}\n`;
        }

        if (data.physicsParams) {
            content += `\n--- Core Physics Parameters ---\n`;
            content += `Gravity (g): ${data.physicsParams.g?.toFixed(2) || 'N/A'} m/s^2\n`;
            content += `Cart Mass: ${data.physicsParams.cart_m?.toFixed(2) || 'N/A'} kg\n`;
            content += `Pendulum 1 Mass (m1): ${data.physicsParams.m1?.toFixed(2) || 'N/A'} kg\n`;
            content += `Pendulum 1 Length (l1): ${data.physicsParams.l1_m?.toFixed(2) || 'N/A'} m\n`;
            content += `Pendulum 2 Mass (m2): ${data.physicsParams.m2?.toFixed(2) || 'N/A'} kg\n`;
            content += `Pendulum 2 Length (l2): ${data.physicsParams.l2_m?.toFixed(2) || 'N/A'} m\n`;
        }

        if (data.physicsRewardConfig) {
            content += `\n--- Reward Weights ---\n`;
            content += `w0: ${data.physicsRewardConfig.w0?.toFixed(3) || 'N/A'}\n`;
            content += `w1: ${data.physicsRewardConfig.w1?.toFixed(3) || 'N/A'}\n`;
            content += `w2: ${data.physicsRewardConfig.w2?.toFixed(3) || 'N/A'}\n`;
            content += `w3: ${data.physicsRewardConfig.w3?.toFixed(3) || 'N/A'}\n`;
            content += `w4: ${data.physicsRewardConfig.w4?.toFixed(3) || 'N/A'}\n`;
            content += `Vp: ${data.physicsRewardConfig.Vp?.toFixed(3) || 'N/A'}\n`;
        }

        if (data.lastStepRewardComponents) {
            content += `\n--- Last Step Reward Breakdown ---\n`;
            content += `Penalty: ${data.lastStepRewardComponents.penalty?.toFixed(3) || 'N/A'}\n`;
            content += `Out of Bounds: ${data.lastStepRewardComponents.outOfBounds?.toFixed(3) || 'N/A'}\n`;
        }


        content += `\n--- UI & System ---\n`;
        content += `Last Action Sent (norm): ${data.lastAction?.toFixed(3) || 'N/A'}\n`;
        content += `Memory Usage: ${data.memoryUsage || '-'}\n`;
        content += `Rendering Visuals: ${isRendering ? 'Yes' : 'No'}\n`;
        panel.textContent = content;
    }

    function updateRewardChartWithDownsampling() {
        if (!rewardChartInstance) return;

        const totalDataPoints = fullRewardHistory.length;
        const labels = [];
        const currentRewards = [];
        const avgRewards = [];
        const bestRewards = [];

        if (totalDataPoints <= MAX_CHART_POINTS) {
            // If we have fewer points than the max, display them all
            fullRewardHistory.forEach(point => {
                labels.push(point.episode);
                currentRewards.push(point.reward);
                avgRewards.push(point.avgReward);
                bestRewards.push(point.bestReward);
            });
        } else {
            // Downsample the data into buckets
            const bucketSize = Math.ceil(totalDataPoints / MAX_CHART_POINTS);
            for (let i = 0; i < totalDataPoints; i += bucketSize) {
                const bucket = fullRewardHistory.slice(i, i + bucketSize);
                if (bucket.length > 0) {
                    const lastPoint = bucket[bucket.length - 1];
                    labels.push(lastPoint.episode);
                    currentRewards.push(lastPoint.reward);
                    avgRewards.push(lastPoint.avgReward);
                    bestRewards.push(lastPoint.bestReward);
                }
            }
        }
        
        rewardChartInstance.data.labels = labels;
        rewardChartInstance.data.datasets[0].data = currentRewards;
        rewardChartInstance.data.datasets[1].data = avgRewards;
        rewardChartInstance.data.datasets[2].data = bestRewards;
        rewardChartInstance.update('none');
    }

    function updateLossChartWithDownsampling() {
        if (!lossChartInstance || !lossChartCtx) return;

        const labels = [];
        const actorLosses = [];
        const criticLosses = [];

        // Downsample actor losses
        const totalActorLossPoints = actorLossHistory.length;
        if (totalActorLossPoints <= MAX_CHART_POINTS) {
            actorLossHistory.forEach(point => {
                // For simplicity, we'll use the episode numbers from actor loss for labels
                // This assumes actor/critic losses are recorded together.
                labels.push(point.episode);
                actorLosses.push(point.loss);
            });
        } else {
            const bucketSize = Math.ceil(totalActorLossPoints / MAX_CHART_POINTS);
            for (let i = 0; i < totalActorLossPoints; i += bucketSize) {
                const bucket = actorLossHistory.slice(i, i + bucketSize);
                if (bucket.length > 0) {
                    const lastPoint = bucket[bucket.length - 1];
                    labels.push(lastPoint.episode); // Use episode from last point in bucket
                    actorLosses.push(lastPoint.loss); // Use loss from last point
                }
            }
        }

        // Downsample critic losses (aligning with actor loss labels for simplicity)
        // This is a simplified alignment; more robust would be to interpolate or ensure same episode counts.
        const totalCriticLossPoints = criticLossHistory.length;
         if (totalCriticLossPoints <= MAX_CHART_POINTS) {
            criticLossHistory.forEach(point => criticLosses.push(point.loss));
        } else {
            const bucketSize = Math.ceil(totalCriticLossPoints / MAX_CHART_POINTS);
            for (let i = 0; i < totalCriticLossPoints; i += bucketSize) {
                const bucket = criticLossHistory.slice(i, i + bucketSize);
                if (bucket.length > 0) criticLosses.push(bucket[bucket.length - 1].loss);
            }
        }
        lossChartInstance.data.labels = labels.slice(0, Math.min(actorLosses.length, criticLosses.length)); // Ensure labels match data length
        lossChartInstance.data.datasets[0].data = actorLosses.slice(0, lossChartInstance.data.labels.length);
        lossChartInstance.data.datasets[1].data = criticLosses.slice(0, lossChartInstance.data.labels.length);
        lossChartInstance.update('none');
    }

    function initRewardChart() {
        if (rewardChartInstance) rewardChartInstance.destroy();
        rewardChartInstance = new Chart(rewardChartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        type: 'scatter', // Change type to scatter for individual episode rewards
                        label: 'Episode Reward', // Clarify label
                        data: [],
                        backgroundColor: 'rgba(96, 165, 250, 0.6)', // Semi-transparent blue for scatter points
                        borderColor: 'rgba(96, 165, 250, 0.9)',    // Border for scatter points
                        pointRadius: 1.5, // Adjust point size for scatter
                        pointHoverRadius: 5
                        // 'fill' and 'tension' are not applicable to scatter type
                    },
                    {
                        label: 'Average Reward (10 episodes)',
                        data: [],
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.1)',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2
                    },
                    {
                        label: 'Best Reward',
                        data: [],
                        borderColor: '#fbbf24',
                        backgroundColor: 'rgba(251, 191, 36, 0.1)',
                        fill: false,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: { 
                        title: { display: true, text: 'Episode', color: '#9ca3af' }, 
                        ticks: { color: '#9ca3af', autoSkip: true, maxTicksLimit: 20 },
                        grid: { color: 'rgba(156, 163, 175, 0.1)' }
                    },
                    y: { 
                        title: { display: true, text: 'Reward', color: '#9ca3af' }, 
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(156, 163, 175, 0.1)' }
                    }
                },
                plugins: { 
                    legend: { 
                        labels: { color: '#e5e7eb' },
                        position: 'top'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(31, 41, 55, 0.9)',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#6b7280',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    function initLossChart() {
        if (!lossChartCtx) return; // Don't initialize if canvas doesn't exist
        if (lossChartInstance) lossChartInstance.destroy();
        lossChartInstance = new Chart(lossChartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Actor Loss',
                        data: [],
                        borderColor: '#f472b6', // Pink
                        backgroundColor: 'rgba(244, 114, 182, 0.1)',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5
                    },
                    {
                        label: 'Critic Loss',
                        data: [],
                        borderColor: '#818cf8', // Indigo
                        backgroundColor: 'rgba(129, 140, 248, 0.1)',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5
                    }
                ]
            },
            options: { // Similar options to reward chart, adjust as needed
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                scales: {
                    x: { title: { display: true, text: 'Episode', color: '#9ca3af' }, ticks: { color: '#9ca3af', autoSkip: true, maxTicksLimit: 20 }, grid: { color: 'rgba(156, 163, 175, 0.1)' }},
                    y: { title: { display: true, text: 'Loss', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.1)' }}
                },
                plugins: {
                    legend: { labels: { color: '#e5e7eb' }, position: 'top' },
                    tooltip: { backgroundColor: 'rgba(31, 41, 55, 0.9)', titleColor: '#e5e7eb', bodyColor: '#e5e7eb', borderColor: '#6b7280', borderWidth: 1 }
                }
            }
        });
    }

    function resetUI() {
        initRewardChart();
        if (lossChartCtx) initLossChart();

        currentChunkRewards = [];
        fullRewardHistory = [];
        actorLossHistory = [];
        criticLossHistory = [];
        // avgLogProbHistory = [];
        chartNeedsUpdate = true; // Force chart update to clear it

        bestRewardEver = -Infinity;
        document.getElementById('episode-counter').textContent = 0;
        document.getElementById('total-reward').textContent = '0.00';
        document.getElementById('best-reward').textContent = '0.00';
        document.getElementById('avg-reward').textContent = '0.00';
        
        // Reset agent-specific UI elements
        document.getElementById('buffer-size').textContent = '0';
        document.getElementById('total-steps').textContent = '0';
        document.getElementById('buffer-progress').textContent = '0/1M';
        document.getElementById('training-status').textContent = 'Warming up...';
        document.getElementById('currentMode').textContent = 'IDLE'; // More accurate for reset state
        document.getElementById('agentState').textContent = 'Ready'; // Or 'Reset'
        document.getElementById('lastAction').textContent = '0.00';
        document.getElementById('training-status').textContent = 'N/A';
        if (spsDisplay) spsDisplay.textContent = '-';

        latestDebugSnapshot = {}; // Clear snapshot
        document.getElementById('memoryUsage').textContent = '-';
        updateDebugInfoPanel(); // Ensure debug panel is cleared/reset
    }
    
    // Set initial text for the repurposed button (user should update HTML text too)
    pauseResumeButton.textContent = 'Reset Pendulum Position';
    pauseResumeButton.disabled = true; // Initially disabled

    // Initial setup
    resetUI();
    worker.postMessage({ type: 'set_speed', payload: { speed: 1 } });
    
    // Check for saved state on load
    const savedState = localStorage.getItem('doublePendulumState');
    if (savedState) {
        loadStateButton.disabled = false;
        showStatusMessage('Saved state detected - use Load State button to restore', 'info');
    }
    
    renderingLoop();

    renderButton.addEventListener('click', () => {
        isRendering = !isRendering;
        renderButton.textContent = isRendering ? 'Pause Visuals' : 'Resume Visuals';
        // Always inform the worker of the new rendering preference.
        // The worker's 'allowRender' flag will ultimately decide if it sends render data.
        worker.postMessage({ type: 'set_render_enabled', payload: { enabled: isRendering } });
    });
    
    // Repurpose pauseResumeButton to resetPendulumButton
    pauseResumeButton.addEventListener('click', () => {
        // This button is now "Reset Pendulum Position"
        worker.postMessage({ type: 'reset_pendulum_physics_state_only' });
    });

    saveStateButton.addEventListener('click', () => {
        worker.postMessage({ type: 'save_state' });
    });
    
    loadStateButton.addEventListener('click', () => {
        const savedState = localStorage.getItem('doublePendulumState');
        if (savedState) {
            try {
                const agentState = JSON.parse(savedState);
                worker.postMessage({ type: 'load_state', payload: { agentState } });
            } catch (error) {
                showStatusMessage('Failed to load saved state: Invalid data', 'error');
            }
        } else {
            showStatusMessage('No saved state found', 'warning');
        }
    });

    speedSlider.addEventListener('input', (e) => {
        const speed = parseInt(e.target.value, 10);
        speedValue.textContent = `x${speed}`;
        worker.postMessage({ type: 'set_speed', payload: { speed } });
    });
    
    if (copyDebugInfoButton) {
        copyDebugInfoButton.addEventListener('click', () => {
            const panel = document.getElementById('debugInfoPanel');
            if (panel && navigator.clipboard) {
                navigator.clipboard.writeText(panel.textContent)
                    .then(() => showStatusMessage('Debug info copied to clipboard!', 'success'))
                    .catch(err => showStatusMessage('Failed to copy debug info: ' + err, 'error'));
            } else {
                showStatusMessage('Clipboard API not available or panel not found.', 'warning');
            }
        });
    }

    trainButton.addEventListener('click', () => {
        if (workerAppMode === 'TRAINING') {
            worker.postMessage({ type: 'pause_simulation' }); // "Pause Training" action
            // UI updates for button text and state will be handled by 'simulation_paused' message
        } else if (workerAppMode === 'PAUSED_TRAINING_SHOWING_POLICY') {
            worker.postMessage({ type: 'resume_simulation' }); // "Resume Training" action
            // UI updates for button text and state will be handled by 'simulation_resumed' message
        } else { 
            // Covers IDLE, OBSERVING, or PAUSED (from OBSERVING)
            // All these states should lead to "Start Training"
            worker.postMessage({ type: 'start_training', payload: { renderEnabled: false } });
            // UI updates for button text and state will be handled by 'training_started' message
            // Visuals off by default for training speed
            isRendering = false; 
            renderButton.textContent = 'Resume Visuals';
            // Ensure worker knows visuals are off for training start
            // (worker's start_training already defaults allowRender based on payload)
        }
    });

    resetButton.addEventListener('click', () => {
        if (confirm('Are you sure you want to reset the agent and all progress?')) {
            // No need to click trainButton first, reset_agent handles stopping current activity
            worker.postMessage({ type: 'reset_agent' });
        }
    });
    
    // Keyboard shortcuts for better UX
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 's':
                    e.preventDefault();
                    if (!saveStateButton.disabled) saveStateButton.click();
                    break;
                case 'l':
                    e.preventDefault();
                    if (!loadStateButton.disabled) loadStateButton.click();
                    break;
                case ' ':
                    // Space bar toggles pause/resume if active, otherwise toggles training/observation
                    // This now primarily targets the trainButton's new Start/Pause/Resume Training cycle
                    e.preventDefault();
                    if (!trainButton.disabled) {
                        trainButton.click();
                    }
                    // The old logic for pauseResumeButton is removed as it's repurposed.
                    // If you want spacebar to also trigger "Reset Pendulum Position":
                    // else if (!pauseResumeButton.disabled) { // Check if resetPendulumButton is active
                    //    pauseResumeButton.click();
                    // }
                    break;
            }
        }
    });
});
