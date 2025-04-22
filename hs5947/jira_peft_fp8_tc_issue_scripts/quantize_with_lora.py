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

# quantize with INC (from measured stats)
os.environ["QUANT_CONFIG"] = "quantization/flux/quantize_config.json"
image = pipe(
    prompt="A picture of sks dog in a bucket",
    quant_mode="quantize",
).images[0]
prefix = "lazy" if is_lazy() else "eager"
image.save(f"{prefix}_dog_quant.png")

# load lora
pipe.load_lora_weights("dsocek/lora-flux-dog", adapter_name="user_lora")

# INC must be done before torch.compile() else it will fail (is this expected behavior?)
if not is_lazy():
    #pipe = torch.compile(pipe, backend="hpu_backend") # <-- whole pipe tc fails for now
    pipe.transformer = torch.compile(pipe.transformer, backend="hpu_backend")
    pipe.vae = torch.compile(pipe.vae, backend="hpu_backend")
    pipe.text_encoder = torch.compile(pipe.text_encoder, backend="hpu_backend")
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, backend="hpu_backend")

# load lora (loading after torch.compile also fails but with different error..)
#pipe.load_lora_weights("dsocek/lora-flux-dog", adapter_name="user_lora")

image = pipe(
    prompt="A picture of sks dog in a bucket",
).images[0]
prefix = "lazy" if is_lazy() else "eager"
image.save(f"{prefix}_dog_quant_lora.png")
