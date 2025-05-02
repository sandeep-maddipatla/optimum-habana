#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
)

# load model
model_name = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
    model_name,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")

outputs = pipe("A cheetah creating a sonic boom")
outputs.images[0].save("out.png")