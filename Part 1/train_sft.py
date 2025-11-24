from transformers import AutoTokenizer, AutoModelForCausalLM
from model_moh import LlamaForCausalLM, LlamaAttention
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
import re

SYSTEM_PROMPT = """
You are a careful, reliable math-reasoning assistant trained to solve grade-school math problems.

You must follow these rules for every answer:
1. Think step by step.
2. Output the final answer on its own line using the exact format, where X is the final answer:

FINAL ANSWER: \\boxed{X}
"""


def process_function(example):
    question = example["question"]
    answer = example["answer"]

    answer = answer.replace("#### ", "FINAL ANSWER: \\boxed{") + "}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages, "question": question, "answer": answer}


def main():
    tokenizer = AutoTokenizer.from_pretrained("/home/mrinal/scratch/layers-to-latents/custom_llama3", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("/home/mrinal/scratch/layers-to-latents/custom_llama3",trust_remote_code=True)
    for name, param in model.named_parameters():
        if "self_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model_to_update = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            model_to_update.append(n)
    params_to_optimize = torch.concat([p.flatten() for n, p in model.named_parameters() if p.requires_grad])
    print("Parameters to optimize: ", params_to_optimize.shape[0])
    print("Optimising: ", model_to_update)

    ds = load_dataset("openai/gsm8k", "main")
    ds["train"] = ds["train"].map(process_function)
    ds["test"] = ds["test"].map(process_function)

    dataset_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir="outputs",
        do_eval=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
        learning_rate=1e-5,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        optim="adamw_torch",
        gradient_accumulation_steps=16,
        eval_strategy="steps",
        eval_steps=100,
    )

    def format_messages(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        formatting_func=format_messages,
    )
    trainer.tokenizer = tokenizer

    trainer.train()

    model.save_pretrained("outputs")
    tokenizer.save_pretrained("outputs")
    print("Training completed....")


if __name__ == "__main__":
    main()