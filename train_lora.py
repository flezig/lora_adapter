import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


# НАСТРОЙКИ
MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_DIR = "outputs/data"
OUTPUT_DIR = "outputs/lora_model"


USE_4BIT = False
MAX_SEQ_LENGTH = 768

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100


def make_prompt(example):
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()

    if input_text:
        text = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}"
        )
    else:
        text = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output}"
        )
    return {"text": text}


def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/val.jsonl",
        },
    )

    dataset = dataset.map(make_prompt)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA-адаптер сохранен в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
