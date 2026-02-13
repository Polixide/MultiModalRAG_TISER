import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from trl import SFTTrainer, SFTConfig
import trackio

from config import (
    QWEN_BASE_ID,
    SFT_OUTPUT_DIR,
    SFT_NUM_EPOCHS,
    SFT_PER_DEVICE_TRAIN_BATCH_SIZE,
    SFT_GRADIENT_ACCUMULATION_STEPS,
    SFT_GRADIENT_CHECKPOINTING,
    SFT_LEARNING_RATE,
    SFT_WARMUP_STEPS,
    SFT_WEIGHT_DECAY,
    SFT_MAX_GRAD_NORM,
    SFT_BF16,
    SFT_FP16,
    SFT_TF32,
    SFT_LR_SCHEDULER_TYPE,
    SFT_LOGGING_STEPS,
    SFT_EVAL_STRATEGY,
    SFT_EVAL_STEPS,
    SFT_SAVE_STRATEGY,
    SFT_SAVE_STEPS,
    SFT_REMOVE_UNUSED_COLUMNS,
    SFT_RUN_NAME,
    SFT_TRACKIO_PROJECT,
    SFT_TRACKIO_SPACE,
)
from modeling.collator_qwen import collate_fn
from modeling.lora_qwen import get_lora_config


def load_base_model(model_id: str = QWEN_BASE_ID):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_sft_config() -> SFTConfig:
    return SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_NUM_EPOCHS,
        per_device_train_batch_size=SFT_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=SFT_GRADIENT_CHECKPOINTING,
        learning_rate=SFT_LEARNING_RATE,
        warmup_steps=SFT_WARMUP_STEPS,
        weight_decay=SFT_WEIGHT_DECAY,
        max_grad_norm=SFT_MAX_GRAD_NORM,
        bf16=SFT_BF16,
        fp16=SFT_FP16,
        tf32=SFT_TF32,
        lr_scheduler_type=SFT_LR_SCHEDULER_TYPE,
        logging_steps=SFT_LOGGING_STEPS,
        report_to="trackio",
        run_name=SFT_RUN_NAME,
        remove_unused_columns=SFT_REMOVE_UNUSED_COLUMNS,
        eval_strategy=SFT_EVAL_STRATEGY,
        eval_steps=SFT_EVAL_STEPS,
        save_strategy=SFT_SAVE_STRATEGY,
        save_steps=SFT_SAVE_STEPS,
    )


def init_tracking(args: SFTConfig):
    trackio.init(
        project=SFT_TRACKIO_PROJECT,
        name=SFT_RUN_NAME,
        config=args,
        space_id=SFT_TRACKIO_SPACE,
        autolog_gpu=True,
    )


def build_sft_trainer(model, processor, train_ds, eval_ds) -> SFTTrainer:
    args = build_sft_config()
    init_tracking(args)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda b: collate_fn(b, processor),
        peft_config=get_lora_config(),
    )
    return trainer


def run_training(trainer: SFTTrainer, resume_from_checkpoint: str | None = None):
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def save_and_push(trainer: SFTTrainer, processor, repo_id: str):
    outdir = trainer.args.output_dir
    trainer.save_model(outdir)
    processor.save_pretrained(outdir)
    trainer.model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
