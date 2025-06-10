import os
import torch
from optimum.habana.diffusers import  GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline

mode = os.environ.get('MODE', 'quant')

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
    use_hpu_graphs=False,
    gaudi_config="Habana/stable-diffusion",
    bf16_full_eval=True,
    torch_dtype=torch.bfloat16
)

if mode == 'measure':
    # dump measure stats through INC
    os.environ["QUANT_CONFIG"] = "quantization/flux/measure_config.json"
    pipe(
        prompt="A picture of sks dog in a bucket",
        quant_mode="measure",
    )
    print('Measurement step done')
elif mode == 'quant':
    # quantize with INC (from measured stats)
    os.environ["QUANT_CONFIG"] = "quantization/flux/quantize_config.json"
    pipe.transformer = torch.compile(pipe.transformer, backend="hpu_backend")
    image = pipe(
        prompt="A picture of sks dog in a bucket",
        quant_mode="quantize"
    ).images[0]
    image.save(f"output_image.png")
    print('Quant Step done')
else:
    print(f'Unrecognized setting for MODE={mode}')
