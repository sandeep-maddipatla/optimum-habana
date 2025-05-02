#!/usr/bin/env python
import torch
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
)
from optimum.habana.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_mlperf import (
        StableDiffusionXLPipeline_HPU,
)
#load model
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = GaudiDDIMScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)
pipe = StableDiffusionXLPipeline_HPU.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)
pipe = torch.compile(pipe, backend="hpu_backend")
outputs = pipe("A cheetah creating a sonic boom")
outputs.images[0].save("out.png")