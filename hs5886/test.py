#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiFluxPipeline,
)
from torch._export import capture_pre_autograd_graph

# load model
model_name = "black-forest-labs/FLUX.1-schnell"
scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

pipe = GaudiFluxPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
input_string = "A picture of a dog in a bucket"
graph = capture_pre_autograd_graph(pipe, (input_string,))
pipe = torch.compile(pipe, backend="hpu_backend")

#outputs = pipe(input_string)
#outputs.images[0].save("out.png")