# Double Inverted Pendulum DDPG

This demo uses a Deep Deterministic Policy Gradient (DDPG) agent implemented with TensorFlow.js and WebAssembly physics to balance a double inverted pendulum.

## Running
1. Build the WebAssembly physics module
   ```
   cargo build --release --target wasm32-unknown-unknown
   ```
2. Serve the `doublepen.html` file from a web server so the worker can load the scripts. Any static file server will work.

The UI shows episode information, reward statistics, and agent configuration while training.
