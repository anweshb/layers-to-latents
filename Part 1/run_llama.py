from datasets import load_dataset
import random
import torch
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import argparse
import yaml
import os
import json
import logging
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

MMLU_MCQ_REGEX = r"Final Answer:\s*([ABCD])"

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset"].get("name", "mmlu")
        self.seed = config.get("seed", 42)
        self.model_name = config["model"].get("name", "Qwen/Qwen3-0.6B")
        self.device = config.get("device", "cuda")
        self.experiment_name = config.get("experiment_name", "mmlu_zero_shot")

        self.setup_logging()
        self.logger.info(f"Config: {self.config}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(self.device).eval()
    
        self.preprocess_dataset()
        self.set_seed(self.seed)
        self.setup_model()
        self.logger.info("Model Architecture: \n\n")
        self.logger.info(self.model)
        self.logger.info("-" * 100)
        self.logger.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.logger.info("-" * 100)
        self.logger.info("\n\n")

    def setup_logging(self):
        os.makedirs(f"results/{self.experiment_name}/logs", exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler(f"results/{self.experiment_name}/logs/run.log")
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(logging.StreamHandler())

    @torch.no_grad()
    def setup_model(self):
        if self.config["model"].get("prune", False):    
            if self.config["model"].get("prune_type", "layers") == "layers":
                layers_to_prune = self.config["model"].get("prune_layers", [])
                layers_to_keep = list(range(self.model.config.num_hidden_layers))
                layers_to_keep = [layer for layer in layers_to_keep if layer not in layers_to_prune]
                self.model.model.layers = torch.nn.ModuleList(
                    [layer for i, layer in enumerate(self.model.model.layers) if i in layers_to_keep]
                )
                self.model.config.num_hidden_layers = len(layers_to_keep)
            elif self.config["model"].get("prune_type", "layers") == "magnitude":
                combined_params = None
                param_combination = []
                for name, param in self.model.named_parameters():
                    values = param.flatten()
                    prev_length = combined_params.shape[0] if combined_params is not None else 0
                    combined_params = torch.cat([combined_params, values], dim=0) if combined_params is not None else values
                    after_length = combined_params.shape[0]
                    param_combination.append((name, prev_length, after_length))

                prune_ratio = self.config["model"].get("prune_percentage", 0.0)
                prune_parameter_count = int(prune_ratio * combined_params.shape[0])
                threshold = torch.kthvalue(torch.abs(combined_params), prune_parameter_count+1).values.item()

                for name, param in self.model.named_parameters():
                    mask = torch.abs(param) < threshold
                    param.masked_fill_(mask, 0.0)

                total_pruned_parameters = 0
                for name, param in self.model.named_parameters():
                    count = (param == 0.0).sum().item()
                    total_pruned_parameters += count
                
                print("Total pruned percentage: ", (total_pruned_parameters / combined_params.shape[0]) * 100)

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def get_prompt_and_label_mmlu(self, row):
        question = row["question"]
        option_a = row["choices"][0]
        option_b = row["choices"][1]
        option_c = row["choices"][2]
        option_d = row["choices"][3]
        answer = ["A", "B", "C", "D"][row["answer"]]

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
        return prompt, answer

    def preprocess_mmlu_data(self, row):
        prompt, answer = self.get_prompt_and_label_mmlu(row)
        messages = [
            {"role": "user", "content": prompt},
        ]
        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
        )
        return prompt, tokenized_inputs, answer

    def find_answer(self, text):
        match = re.search(r"#### \d+", text)
        if match:
            return match.group(0).replace("#### ", "")
        else:
            return None

    def preprocess_dataset(self):
        if self.dataset_name == "mmlu":
            ds = load_dataset("cais/mmlu", "all")
            shuffled_ds = ds[self.config["dataset"].get("subset", "test")].shuffle(seed=self.config.get("seed", 42))
            subset_data = shuffled_ds.select(range(int(len(shuffled_ds) * self.config["dataset"].get("subset_size", 0.1))))
            self.dataset = subset_data
            self.logger.info(f"Processing of MMLU dataset completed with {len(self.dataset)} examples...")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

    def postprocess_mmlu_response(self, outputs):
        match = re.search(r"\\boxed{(\w)}", outputs)
        if match:
            return match.group(1)
        else:
            match = re.search(r"The final answer is (\w)", outputs)
            if match:
                return match.group(1)
            else:
                return None

    @torch.no_grad()
    def evaluate(self):
        ground_truth = []
        predictions = []
        queries = []
        for row in tqdm(self.dataset, desc="Evaluating dataset"):
            if self.dataset_name == "mmlu":
                prompt, tokenized_inputs, answer = self.preprocess_mmlu_data(row)
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported")
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

            if self.dataset_name == "mmlu":
                with torch.no_grad():
                    outputs = self.model.generate(**tokenized_inputs, max_new_tokens=2048, use_cache=False, temperature=0.3, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.2)
                decoded_outputs = self.tokenizer.decode(outputs[:, tokenized_inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
                final_answer = self.postprocess_mmlu_response(decoded_outputs)
                print("Outputs: ", final_answer, "Answer: ", answer)
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported")
            if self.dataset_name == "mmlu":
                queries.append(row["question"])
                if final_answer is None:
                    continue
                ground_truth.append(answer)
                predictions.append(final_answer)
            else:
                queries.append(row["question"])
                ground_truth.append(int(answer))
                predictions.append(final_answer)

        if self.dataset_name == "mmlu":
            accuracy = accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, average='macro', labels=["A", "B", "C", "D"], zero_division=0)
            recall = recall_score(ground_truth, predictions, average='macro', labels=["A", "B", "C", "D"], zero_division=0)
            f1 = f1_score(ground_truth, predictions, average='macro', labels=["A", "B", "C", "D"], zero_division=0)
            report = classification_report(ground_truth, predictions, labels=["A", "B", "C", "D"], zero_division=0)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        responses = []
        for query, gt, pred in zip(queries, ground_truth, predictions):
            responses.append({"query": query, "ground_truth": gt, "prediction": pred})

        with open(f"results/{self.experiment_name}/responses.json", "w") as f:
            json.dump({"responses": responses}, f)

        self.logger.info("Evaluation summary: ")
        self.logger.info(f"Accuracy: {accuracy}")
        if self.dataset_name == "mmlu":
            self.logger.info(f"Precision: {precision}")
            self.logger.info(f"Recall: {recall}")
            self.logger.info(f"F1: {f1}")
            self.logger.info("Classification report:\n" + report)
        self.logger.info("--------------------------------")
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True)
        if self.dataset_name == "mmlu":
            with open(f"results/{self.experiment_name}/evaluation_results.json", "w") as f:
                json.dump({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "report": report}, f)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()