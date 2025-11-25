from transformers import AutoTokenizer, AutoModelForCausalLM
from model_moh import LlamaForCausalLM, LlamaAttention
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
import re

def process_function(example):
    question = example["question"]
    option_a = example["choices"][0]
    option_b = example["choices"][1]
    option_c = example["choices"][2]
    option_d = example["choices"][3]
    answer = ["A", "B", "C", "D"][example["answer"]]

    prompt = (
        "You are given a multiple-choice question. Choose the correct option.\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"C. {option_c}\n"
        f"D. {option_d}\n\n"
        'Your answer must end with: "The final answer is \\boxed{X}" where X is the correct letter.'
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "The final answer is \\boxed{" + answer + "}"},
    ]
    return {"messages": messages, "question": question, "answer": answer}


def main():
    tokenizer = AutoTokenizer.from_pretrained("custom_llama3", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("custom_llama3",trust_remote_code=True)
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

    ds = load_dataset("cais/mmlu", "all")
    ds["auxiliary_train"] = ds["auxiliary_train"].map(process_function)
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
        learning_rate=3e-4,
        num_train_epochs=1,
        per_device_train_batch_size=1,
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
        train_dataset=ds["auxiliary_train"],
        eval_dataset=ds["test"],
        formatting_func=format_messages,
    )
    trainer.tokenizer = tokenizer

    trainer.train()

    model.save_pretrained("outputs_mmlu")
    tokenizer.save_pretrained("outputs_mmlu")
    print("Training completed....")


if __name__ == "__main__":
    main()