#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiStableDiffusionPipeline,
)

# load model
model_name = "stabilityai/stable-diffusion-2-1"
scheduler = GaudiDDIMScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

pipe = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")

outputs = pipe("A picture of an elephant flying over the clouds")
outputs.images[0].save("out.png")