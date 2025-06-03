import torch
from torch._dynamo.minifier import Minifier

from optimum.habana import GaudiConfig
from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.transformers.trainer import _is_peft_model
from optimum.habana.utils import set_seed
from contextlib import nullcontext

import warnings
warnings.filterwarnings("ignore")

from train_dreambooth import create_text_encoder_adapter_config, create_unet_adapter_config, parse_args, b2mb, DreamBoothDataset,  collate_fn, PromptDataset

def main()
    logging_dir = Path(args.output_dir, args.logging_dir)

    gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
    gaudi_config.use_torch_autocast = gaudi_config.use_torch_autocast or args.mixed_precision == "bf16"
    accelerator = GaudiAccelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        force_autocast=gaudi_config.use_torch_autocast,
    )
    if args.report_to == "wandb":
        import wandb

        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.sdp_on_bf16:
        torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            api = HfApi(token=args.hub_token)
            # Create repo (repo_name from args or inferred)
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    logger.info(f'Text encoder class import done')

    # Load scheduler and models
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    if args.adapter != "full":
        config = create_unet_adapter_config(args)
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()
    unet.to(accelerator.device)
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    elif args.train_text_encoder and args.adapter != "full":
        config = create_text_encoder_adapter_config(args)
        text_encoder = get_peft_model(text_encoder, config)
        text_encoder.print_trainable_parameters()

    torch._dynamo.allow_in_graph(torch.hpu.synchronize)
    unet = torch.compile(unet, backend="hpu_backend")
    logger.info(f'torch.compile called on unet')

    text_encoder.to(accelerator.device)
    logger.info(f'Text encoder moved to device')


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder and not args.adapter != "full":
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if gaudi_config.use_fused_adam:
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW

        optimizer_class = FusedAdamW
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=1,
    )
    logger.info(f'Train_Dataloader initiated')

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def unwrap_model(model, training=False):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        if not training:
            return model
        else:
            if accelerator.distributed_type == DistributedType.MULTI_HPU:
                kwargs = {}
                kwargs["gradient_as_bucket_view"] = True
                accelerator.ddp_h0andler = DistributedDataParallelKwargs(**kwargs)
            if args.use_hpu_graphs_for_training:
                if _is_peft_model(model):
                    base_model = model.get_base_model()
                    htcore.hpu.ModuleCacher()(model=base_model, inplace=True)
                else:
                    htcore.hpu.ModuleCacher()(model=model, inplace=True)
            return model

    unwrap_model(model=unet, training=True)
    if args.train_text_encoder:
        unwrap_model(model=text_encoder, training=True)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    epoch_count = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f'==========={epoch_count=}===========')
        unet.train()
        logger.info(f'Unet.train done')

        if args.train_text_encoder:
            text_encoder.train()
            logger.info(f'text_encoder.train done')

        enable_profile = True
        profiler_ctx =  torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU] if enable_profile else [torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
            profile_memory=enable_profile,
            record_shapes=enable_profile
            )
        ctx = profiler_ctx if enable_profile else nullcontext()
        with ctx as profiler:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb":
                            accelerator.print(progress_bar)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    #unet_result = unet(noisy_latents, timesteps, encoder_hidden_states)
                    Minifier(unet, [noisy_latents, timesteps, encoder_hidden_states], 'hpu_backend').run()
                    model_pred = unet_result.sample
                    logger.info(f'Unet call with noisy_latents done')


                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    logger.info(f'Loss compute done')

                    torch.hpu.synchronize()
                    accelerator.backward(loss)

                    logger.info(f'accelerator Backward step done')
                    htcore.mark_step()
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    htcore.mark_step()
                    logger.info(f'optimizer step and zero grad done')


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb":
                        accelerator.print(progress_bar)
                    global_step += 1

                    # if global_step % args.checkpointing_steps == 0:
                    #     if accelerator.is_main_process:
                    #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    #         accelerator.save_state(save_path)
                    #         logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (
                    args.validation_prompt is not None
                    and (step + num_update_steps_per_epoch * epoch) % args.validation_steps == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        revision=args.revision,
                        use_hpu_graphs=args.use_hpu_graphs_for_inference,
                        use_habana=True,
                        gaudi_config=gaudi_config,
                    )
                    # set `keep_fp32_wrapper` to True because we do not want to remove
                    # mixed precision hooks while we are still training
                    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # Set evaliation mode
                    pipeline.unet.eval()
                    pipeline.text_encoder.eval()

                    # run inference
                    if args.seed is not None:
                        if accelerator.device == torch.device("hpu"):
                            # torch.Generator() is unsupported on HPU
                            generator = set_seed(args.seed)
                        else:
                            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    else:
                        generator = None
                    images = []
                    for _ in range(args.num_validation_images):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                        images.append(image)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            import wandb

                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    # Set evaliation mode
                    pipeline.unet.train()
                    if args.train_text_encoder:
                        pipeline.text_encoder.train()

                    del pipeline

                profiler.step()

                if global_step >= args.max_train_steps:
                    break

        print(profiler.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        print(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))
    
    # Create the pipeline using using the trained modules and save it.
    logger.info(f'Waiting for everyone')

    accelerator.wait_for_everyone()
    logger.info(f'Done waiting')

    if accelerator.is_main_process:
        if args.adapter != "full":
            unwarpped_unet = unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if args.train_text_encoder:
                unwarpped_text_encoder = unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrap_model(unet),
                text_encoder=unwrap_model(text_encoder),
                revision=args.revision,
                use_hpu_graphs=args.use_hpu_graphs_for_inference,
                use_habana=True,
                gaudi_config=gaudi_config,
            )
            pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                run_as_future=True,
            )

    logger.info(f'Before End Training')
    accelerator.end_training()
    logger.info(f'Ended Training')

if __name__ == "__main__":
    args = parse_args()
    main(args)
