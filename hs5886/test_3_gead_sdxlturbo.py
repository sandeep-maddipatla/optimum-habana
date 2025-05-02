#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerAncestralDiscreteScheduler,
)
#load model
model_name = "stabilityai/sdxl-turbo"
scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)
pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")
outputs = pipe("A dog chasing a tiger")
outputs.images[0].save("out.png")