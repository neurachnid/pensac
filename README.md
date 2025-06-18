# Double Inverted Pendulum DDPG

This demo uses a Deep Deterministic Policy Gradient (DDPG) agent implemented with TensorFlow.js and WebAssembly physics to balance a double inverted pendulum.

## Running
1. Build the WebAssembly physics module
   ```
   cargo build --release --target wasm32-unknown-unknown
   ```
2. Serve the `doublepen.html` file from a web server so the worker can load the scripts. Any static file server will work.

The UI shows episode information, reward statistics, and agent configuration while training.

Physics and reward parameters follow the reference DDPG paper (cart mass 0.350 kg, pendulum masses 0.133/0.025 kg, lengths 0.5 m, damping 0.05/0.001/0.001).
The reward weights are w0=0.1, w1=5, w2=5, w3=1, w4=0.05 with an out-of-bounds penalty Vp=100.
