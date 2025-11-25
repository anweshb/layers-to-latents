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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
    
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
        subject = row["subject"]
        choices = row["choices"]
        answer = row["answer"]

        prompt = f"""

        Answer the following multiple choice question by giving the most appropriate response. Answer should be a single character among [A, B, C, D]. 

        Question: {question} ?
            A. {choices[0]}
            B. {choices[1]}
            C. {choices[2]}
            D. {choices[3]}

        Answer:"""
        return prompt, answer

    def get_prompt_and_label_gsm8k(self, row):
        question = row["question"]
        answer = row["response"]
        prompt = """
        You are a helpful math reasoning assistant. Solve the following grade-school math problems. Please reason step by step, and put your final answer within \\boxed{}.

        ### Example 1
        Q: Jake bought 3 packs of markers. Each pack has 12 markers. He gave 5 markers to his friend. How many markers does Jake have now?

        Let's think step by step.
        He bought 3 packs × 12 markers = 36 markers.
        He gave away 5 markers, so 36 − 5 = 31.
        The answer is \\boxed{31}.

        ### Example 2
        Q: A farmer has 48 apples. He wants to pack them into boxes, each holding 6 apples. How many boxes can he fill?

        Let’s think step by step.
        Each box holds 6 apples.  
        48 ÷ 6 = 8 boxes.
        The answer is \\boxed{8}.

        ### Example 3
        Q: Maria read 15 pages on Monday, twice as many on Tuesday, and 10 fewer pages on Wednesday than on Tuesday. How many pages did she read in total?

        Let’s think step by step.
        Tuesday pages = 2 × 15 = 30.
        Wednesday pages = 30 − 10 = 20.
        Total = 15 + 30 + 20 = 65.
        The answer is \\boxed{65}.

        ### Example 4
        Q: A toy costs $14. Emma buys 4 of them and pays with a $100 bill. How much change does she get?

        Let’s think step by step.
        Cost of 4 toys = 4 × 14 = 56.
        Change = 100 − 56 = 44.
        The answer is \\boxed{44}.

        ### Example 5
        Q: A school has 280 students. 3/4 of them take the bus. How many students take the bus?

        Let’s think step by step.
        3/4 of 280 = 280 × 3/4 = 210.
        The answer is \\boxed{210}.

        Q: <<QUESTION>>

        Let’s think step by step.
        """
        return prompt.replace("<<QUESTION>>", question), answer


    def preprocess_mmlu_data(self, row):
        prompt, answer = self.get_prompt_and_label_mmlu(row)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Q: " + row["question"]},
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


    def preprocess_gsm8k_data(self, row):
        row["response"] = self.find_answer(row["answer"])
        prompt, answer = self.get_prompt_and_label_gsm8k(row)
        messages = [
            {"role": "user", "content": prompt, "thinking_budget": 1024},
        ]
        
        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True
        )
        return prompt, tokenized_inputs, answer

    def preprocess_dataset(self):
        if self.dataset_name == "mmlu":
            ds = load_dataset("cais/mmlu", "all")
            shuffled_ds = ds[self.config["dataset"].get("subset", "test")].shuffle(seed=self.config.get("seed", 42))
            subset_data = shuffled_ds.select(range(int(len(shuffled_ds) * self.config["dataset"].get("subset_size", 0.1))))
            self.dataset = subset_data
            self.logger.info(f"Processing of MMLU dataset completed with {len(self.dataset)} examples...")
        elif self.dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main")
            shuffled_ds = ds[self.config["dataset"].get("subset", "test")].shuffle(seed=self.config.get("seed", 42))
            subset_data = shuffled_ds.select(range(int(len(shuffled_ds) * self.config["dataset"].get("subset_size", 0.1))))
            self.dataset = subset_data
            self.logger.info(f"Processing of GSM8K dataset completed with {len(self.dataset)} examples...")
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

    def postprocess_mmlu_response(self, outputs):
        probability = torch.softmax(outputs.logits[:, -1, :], dim=-1)
        selection_ids = self.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
        selection_probability = probability[0, selection_ids]
        selection_index = torch.argmax(selection_probability).item()
        return selection_index

    def postprocess_gsm8k_response(self, decoded_outputs):
        print(decoded_outputs)
        decoded_outputs = re.sub(r"<think>.*?</think>", " ", decoded_outputs, flags=re.DOTALL)
        match = re.search(r"\\boxed{(\d+)}", decoded_outputs)
        if match:
            return int(match.group(1))
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
            elif self.dataset_name == "gsm8k":
                prompt, tokenized_inputs, answer = self.preprocess_gsm8k_data(row)
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported")
            tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

            if self.dataset_name == "mmlu":
                with torch.no_grad():
                    outputs = self.model(**tokenized_inputs)
                final_answer = self.postprocess_mmlu_response(outputs)
            elif self.dataset_name == "gsm8k":
                with torch.no_grad():
                    outputs = self.model.generate(**tokenized_inputs, max_new_tokens=2048, use_cache=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
                decoded_outputs = self.tokenizer.decode(outputs[:, tokenized_inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
                final_answer = self.postprocess_gsm8k_response(decoded_outputs)
                print("Output answer: ", answer)
                print("Question: ", row["question"])
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported")
            queries.append(row["question"])
            ground_truth.append(int(answer))
            predictions.append(final_answer)

        if self.dataset_name == "mmlu":
            accuracy = accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, average='macro', labels=[0, 1, 2, 3], zero_division=0)
            recall = recall_score(ground_truth, predictions, average='macro', labels=[0, 1, 2, 3], zero_division=0)
            f1 = f1_score(ground_truth, predictions, average='macro', labels=[0, 1, 2, 3], zero_division=0)
            report = classification_report(ground_truth, predictions, labels=[0, 1, 2, 3], zero_division=0)
        elif self.dataset_name == "gsm8k":
            count = 0
            for gt, pred in zip(ground_truth, predictions):
                if gt == pred:
                    count += 1
            accuracy = count/len(ground_truth)
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
        elif self.dataset_name == "gsm8k":
            with open(f"results/{self.experiment_name}/evaluation_results.json", "w") as f:
                json.dump({"accuracy": accuracy}, f)
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