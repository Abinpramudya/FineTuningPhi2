import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    TaskType
)

# === Print device availability ===
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# === Disable W&B tracking (optional) ===
os.environ["WANDB_DISABLED"] = "true"

# === Load dataset ===
train_data = load_dataset('json', data_files='split_output/train.jsonl', split='train')
eval_data = load_dataset('json', data_files='split_output/valid.jsonl', split='train')

# === Format prompts ===
def format_prompt(example):
    return f"### Prompt: {example['prompt']}\n### Response: {example['output']}"

# === Tokenizer ===
base_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# === Tokenization function ===
max_length =70
def tokenize(example):
    out = tokenizer(
        format_prompt(example),
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_train = train_data.map(tokenize)
tokenized_eval = eval_data.map(tokenize)

# === Load full model (no quantization) ===
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,  # use GPU-friendly float16
    trust_remote_code=True
)

# === Prepare for LoRA ===
model.config.use_cache = False
model = prepare_model_for_int8_training(model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./output_phi2_lora",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,  # this triggers GPU usage and mixed precision
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer
)

# === Debug confirmation before training ===
print("Trainer device:", trainer.args.device)
print("Model final device:", next(trainer.model.parameters()).device)

# === Train ===
trainer.train()

# === Save output ===
model.save_pretrained("output_phi2_lora")
tokenizer.save_pretrained("output_phi2_lora")