import torch
import torch.nn as nn
import numpy as np
import json
import os
from simple_nn import set_seed,model
set_seed(43)

# run model once to get the graph
fixed_values = [112, 212, 134, 4, 43, 3126, 7, 8321, 1, 10]
fixed_input_tensor = torch.tensor([fixed_values], dtype=torch.float32)
scripted = torch.jit.trace(model, fixed_input_tensor)

# Export folder
OUT_DIR = "web_model"
os.makedirs(OUT_DIR, exist_ok=True)

# Map names to modules
name_to_module = dict(model.named_modules())
layers = []
weights_dict = {}

# Walk graph
for node in scripted.graph.nodes():
    if node.kind() == "prim::CallMethod":
        target = node.inputsAt(0).debugName()   # e.g. %fc1
        method = node.s("name")                 # "forward"

        if target in name_to_module:
            mod = name_to_module[target]

            if isinstance(mod, nn.Linear):
                # Save weights & biases
                w_name = f"{target}_weight.npy"
                b_name = f"{target}_bias.npy"

                np.save(os.path.join(OUT_DIR, w_name), 
                        mod.weight.detach().cpu().numpy().astype(np.float32))
                np.save(os.path.join(OUT_DIR, b_name), 
                        mod.bias.detach().cpu().numpy().astype(np.float32))

                weights_dict[f"{target}.weight"] = w_name
                weights_dict[f"{target}.bias"] = b_name

                layers.append({
                    "name": target,
                    "type": "linear",
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "weight": w_name,
                    "bias": b_name
                })

            elif isinstance(mod, nn.ReLU):
                layers.append({
                    "name": target,
                    "type": "relu"
                })

with open(os.path.join(OUT_DIR, "graph.json"), "w") as f:
    json.dump(layers, f, indent=2)

with open(os.path.join(OUT_DIR, "weights.json"), "w") as f:
    json.dump(weights_dict, f, indent=2)

print("✅ Export done! Graph + weights in", OUT_DIR)
