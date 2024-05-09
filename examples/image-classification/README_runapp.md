# Using run_app.sh script

[run_app.sh] script is provided to launch the image-classification training sample based on the HFTransformers library. The actual model and dataset used are configurable. This document attempts to capture the relevant command lines.

### Supported controls

- MODEL : huggingface model id (default: timm/resnetv2_50x1_bit.goog_in21k)
- DATASET : huggingface dataset id (default: cifar10)
- EPOCHS : Number of training epochs (default: 5).  Strongly recommended to use lower number (e.g. 1 if run on CPUs)
- OUTDIR: Location for outputs storage (default: /tmp/outputs)
- ICNAME: Col name in dataset for images (default: img - corresponds to cifar10). imagenet-1k needs to use "image"

Accepts optional argument to specify URL or path to preprocessor config file.

# Common Examples

## Default run (cifar10 dataset)

Uses cifar10 dataset.

```
EPOCHS=1 ./run_app.sh
```

## Use imagenet-1k dataset from hf-hub

> [!NOTE]
> * This dataset needs env where we are logged in to HF-Hub. One-time setup instructions are at bottom of this doc.
> * This dataset requires approximately 30GB+ disk space. If not already present on system, first run will download this and take a few minutes to finish.

Login to HF-hub:
```
huggingface-cli login --token InsertTokenInClrTxtHere
```

Run with command below:
```
EPOCHS=1 DATASET=imagenet-1k ICNAME=image ./run_app.sh 
```

## Specify alternate config

This is meant to be used by the HF Transformers library auto classes. Specifically the `AutoImageProcessor.from_pretrained` function.

By default the model hub is expected by above function to have a `preprocessor_config.json` file. However, if a model repo (as in the case of the TIMM Resnet models) lacks one, the function allows specification of custom local path or URL for such a file. A local file is provided in this directory. For more info, refer to comments in [GS-123](https://habana.atlassian.net/browse/GS-123)

To use this, simply specify as a command line arg to `run_app.sh`:

```
# Specify URL
./run_app.sh https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k/raw/main/config.json

# Specify Local path
./run_app.sh $(pwd)/preprocessor_config.json
```

# Create and Use HuggingFace Hub Login

- Sign up using your email from [the signup page](https://huggingface.co/join). Needs email confirmation.
- Log-in using the account from [the login page](https://huggingface.co/login)
- Once logged in, create an access token from [here](https://huggingface.co/settings/tokens)33

On the target machine (or docker container), install HuggingFace Hub tools:

```
# Use exactly as below (including the bracketed part)
pip install -U "huggingface_hub[cli]"

# Help command to confirm successful installation
huggingface-cli --help
```


Login to HF-hub:

```
huggingface-cli login --token InsertTokenInClrTxtHere
```
