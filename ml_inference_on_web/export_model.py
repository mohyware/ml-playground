import torch
import torch.nn as nn
import numpy as np
import json
import os
from simple_nn import  set_seed,model

set_seed(43)

OUT_DIR = "web_model"
OS_MAKE_DIRS = os.makedirs(OUT_DIR, exist_ok=True)
layers = []
weights_dict = {}

for name, module in model.named_children():
    layer_info = {"name": name}
    if isinstance(module, nn.Linear):
        w_name = f"{name}_weight"
        b_name = f"{name}_bias"
        np.save(f"{OUT_DIR}/{w_name}.npy", module.weight.detach().cpu().numpy().astype(np.float32))
        np.save(f"{OUT_DIR}/{b_name}.npy", module.bias.detach().cpu().numpy().astype(np.float32))
        weights_dict[f"{name}.weight"] = f"{w_name}.npy"
        weights_dict[f"{name}.bias"] = f"{b_name}.npy"

        layer_info.update({
            "type": "linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weight": f"{w_name}.npy",
            "bias": f"{b_name}.npy"
        })
    elif isinstance(module, nn.ReLU):
        layer_info["type"] = "relu"
    layers.append(layer_info)

# Save network spec and weight mapping
with open(f"{OUT_DIR}/graph.json", "w") as f:
    json.dump(layers, f, indent=2)
with open(f"{OUT_DIR}/weights.json", "w") as f:
    json.dump(weights_dict, f, indent=2)

print("Network JSON generated!")
