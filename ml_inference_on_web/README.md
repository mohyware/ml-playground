## Overview
Experiment to run simple neural network inference on the web with WebGPU.

SimpleNet takes a 10-feature input, passes it through a hidden layer with 16 neurons and a ReLU non-linearity, then outputs 3 values representing the scores (logits) for 3 possible classes.

**Architecture:** input (10 features) → hidden (16 with ReLU) → output (3 classes)

*Note: A seed is used to reproduce the same results and verify the model works correctly on the web.*

## Steps
1. Run `export_model.py` to get the network and weights
2. Run `index.html` with a server (use Chrome and enable WebGPU)
3. Results should match the Python code (`simple_nn` file) - change inputs on both to verify 