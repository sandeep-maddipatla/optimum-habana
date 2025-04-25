#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiEulerDiscreteScheduler,
    GaudiStableDiffusionXLPipeline,
)

# load model
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
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

outputs = pipe("A picture of a dog in a bucket")
outputs.images[0].save("out.png")