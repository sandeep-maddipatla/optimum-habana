#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiFluxPipeline,
)
#load model
model_name = "black-forest-labs/FLUX.1-schnell"

pipe = GaudiFluxPipeline.from_pretrained(
    model_name,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")
outputs = pipe("A picture of a dog in a bucket")
outputs.images[0].save("out.png")