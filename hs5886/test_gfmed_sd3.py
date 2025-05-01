#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiStableDiffusion3Pipeline,
    GaudiFlowMatchEulerDiscreteScheduler,
)
#load model
model_name = "stabilityai/stable-diffusion-3.5-medium"
scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)
pipe = GaudiStableDiffusion3Pipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")
outputs = pipe("A dog chasing a car")
outputs.images[0].save("out.png")