#!/usr/bin/env python
import os
import torch
from optimum.habana.diffusers import (
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiFluxPipeline,
)
from habana_frameworks.torch.utils.internal import is_lazy

# load model
model_name = "black-forest-labs/FLUX.1-dev"
scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)
pipe = GaudiFluxPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True if is_lazy() else False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)

# quantize - measure stats
os.environ["QUANT_CONFIG"] = "quantization/flux/measure_config.json"
pipe(
    prompt="A picture of sks dog in a bucket",
    quant_mode="measure",
)
print("Successfully measured stats for INC quantization")
