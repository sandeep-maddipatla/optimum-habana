#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Copied from https://huggingface.co/docs/transformers/model_doc/owlvit

import argparse
import time

import habana_frameworks.torch as ht
import requests
import torch
from PIL import Image

from transformers import AutoConfig, AutoProcessor, AutoModelForObjectDetection, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import load_dataset

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:
    def check_optimum_habana_min_version(*a, **b):
        return ()

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers and Optimum Habana are not installed. Remove at your own risks.
check_min_version("4.38.0")
check_optimum_habana_min_version("1.10.0")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    image_path: str = field(
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        metadata={"help": 'Path of the input image. Should be a single string (eg: --image_path "URL").'},
    )
    print_result: bool = field(
        default=False,
        metadata={"help" : "Whether to print the classification results."},
    )
    precision: str = field(
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        metadata={"help" : "Specify precision to run model with"}
    )

def main():
    use_habana = False
    try:
        # Try to capture for CPU mode
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))       else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    except ValueError as e:
        if "--use_habana" in str(e):
            use_habana = True
            parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiTrainingArguments))
            if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
                # If we pass only one argument to the script and it's the path to a json file,
                # let's parse it to get our arguments.
                model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
            else:
                model_args, data_args, training_args = parser.parse_args_into_dataclasses()
                                                                
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    gaudi_config = GaudiConfig.from_pretrained(
        training_args.gaudi_config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )


    # Log on each process the small summary:
    mixed_precision = training_args.bf16 or gaudi_config.use_torch_autocast
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        + f"mixed-precision training: {mixed_precision}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name_or_path,
        config=config
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path
    )

    image = Image.open(requests.get(args.image_path, stream=True).raw)
 
    if "hpu" in args.device and args.use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)
        
    autocast = handle_autocast(args)
    model.to(args.device)

    with torch.no_grad(), autocast:
        for i in range(args.warmup):
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            __synchronize(args.device)

        total_model_time = 0
        for i in range(args.n_iterations):
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            __synchronize(args.device)
            model_start_time = time.time()
            outputs = model(**inputs)
            __synchronize(args.device)

            model_end_time = time.time()
            total_model_time = total_model_time + (model_end_time - model_start_time)

            if args.print_result:
                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                target_sizes = torch.Tensor([image.size[::-1]])

                # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                results = processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.1
                )
                if i == 0:
                    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
                    for box, score, label in zip(boxes, scores, labels):
                        box = [round(i, 2) for i in box.tolist()]
                        print(f"Detected label={label} with confidence {round(score.item(), 3)} at location {box}")

    print("n_iterations: " + str(args.n_iterations))
    print("Total latency (ms): " + str(total_model_time * 1000))
    print("Average latency (ms): " + str(total_model_time * 1000 / args.n_iterations))


if __name__ == "__main__":
    main()
