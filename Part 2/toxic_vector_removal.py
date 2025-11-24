import torch
import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from toxic_classification_gpt2 import GPT2Activations, ToxicCommentDataset, HarmfulDetectorMLP

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HarmfulVectorConstructor:
    def __init__(self, activation_dir):
        self.activation_dir = activation_dir

    def get_mean_difference_vector(self, layer_idx):
        data = np.load(os.path.join(self.activation_dir, f'train_layer_{layer_idx}_pooled_activations.npy'))
        labels = np.load(os.path.join(self.activation_dir, 'train_labels.npy'))
        
        toxic_mean = np.mean(data[labels==1], axis=0)
        non_toxic_mean = np.mean(data[labels==0], axis=0)
        vector = toxic_mean - non_toxic_mean
        return vector / np.linalg.norm(vector)

    def get_logistic_vector(self, layer_idx):
        data = np.load(os.path.join(self.activation_dir, f'train_layer_{layer_idx}_pooled_activations.npy'))
        labels = np.load(os.path.join(self.activation_dir, 'train_labels.npy'))
        
        clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        clf.fit(data, labels)
        
        vector = clf.coef_[0]
        return vector / np.linalg.norm(vector)

class SafetyHookManager:
    def __init__(self, model, device):
        self.model = model
        self.hooks = []
        self.device = device
    
    def register_hooks(self, layer_vectors, alpha):
        self.clear_hooks()
        
        for layer_idx, vector in layer_vectors.items():
            v = torch.tensor(vector, dtype=torch.float32, device=self.device)
            hook_fn = self._get_hook_fn(v, alpha)
            
            if 0 <= layer_idx < len(self.model.h):
                module = self.model.h[layer_idx]
                self.hooks.append(module.register_forward_hook(hook_fn))
                
    def _get_hook_fn(self, harmful_vector, alpha):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            v = harmful_vector.to(hidden_states.device).to(hidden_states.dtype)
            
            projections = torch.matmul(hidden_states, v)
            modified_hidden = hidden_states - alpha * projections.unsqueeze(-1) * v
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def evaluate_safety(gpt2_extractor, test_loader, hook_manager, layer_vectors, alpha, judge_mlp, judge_layer_idx):
    if alpha > 0:
        hook_manager.register_hooks(layer_vectors, alpha)
    else:
        hook_manager.clear_hooks()
    
    all_preds = []
    all_labels = []
    
    gpt2_extractor.model.eval()
    judge_mlp.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval alpha={alpha}", leave=False):
            input_ids = batch['input_ids'].to(gpt2_extractor.device)
            attention_mask = batch['attention_mask'].to(gpt2_extractor.device)
            labels = batch['label'].numpy()
            
            outputs = gpt2_extractor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            judge_activations = outputs.hidden_states[judge_layer_idx + 1]
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(judge_activations.size()).float()
            sum_embeddings = torch.sum(judge_activations * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_activations = sum_embeddings / sum_mask
            
            mlp_out = judge_mlp(pooled_activations)
            preds = (mlp_out > 0.5).cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    hook_manager.clear_hooks()
    
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return f1, acc, fpr, tpr

def main():
    set_seed(42)
    
    SAVE_DIR = '/home/anwesh/scratch/layers-to-latents/gpt2_activations'
    RESULTS_CSV = os.path.join(SAVE_DIR, 'results.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        results_df = results_df.sort_values('test_f1', ascending=False)
        best_layer_idx = int(results_df.iloc[0]['layer_idx'])
        print(f"Best performing layer is: {best_layer_idx}")
    else:
        raise FileNotFoundError(f"Results file not found at {RESULTS_CSV}")

    torch.cuda.empty_cache()
    gc.collect()

    print("Initializing GPT-2 and DataLoader...")
    BATCH_SIZE = 32
    gpt2_extractor = GPT2Activations()
    test_dataset = ToxicCommentDataset(split='test', tokenizer=gpt2_extractor.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading Judge MLP for Layer {best_layer_idx}...")
    judge_mlp = HarmfulDetectorMLP(input_size=gpt2_extractor.h_size).to(DEVICE)
    judge_mlp.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'mlp_layer_{best_layer_idx}_model.pth')))

    alphas = [0.0, 0.5, 1.0]
    layer_counts = [1, 3, 5, 10]
    methods = ['mean_diff', 'logistic']

    results = []
    constructor = HarmfulVectorConstructor(SAVE_DIR)
    hook_manager = SafetyHookManager(gpt2_extractor.model, DEVICE)

    print("\nStarting Safety Alignment Experiments...")
    print("="*80)

    for method in methods:
        print(f"\nMethod: {method.upper()}")
        
        for n_layers in layer_counts:
            top_layers = results_df.head(n_layers)['layer_idx'].values.astype(int)
            print(f"  Editing Top {n_layers} Layers: {top_layers}")
            
            layer_vectors = {}
            for layer_idx in top_layers:
                if method == 'mean_diff':
                    vec = constructor.get_mean_difference_vector(layer_idx)
                else:
                    vec = constructor.get_logistic_vector(layer_idx)
                layer_vectors[layer_idx] = vec
                
            for alpha in alphas:
                torch.cuda.empty_cache()
                
                f1, acc, fpr, tpr = evaluate_safety(
                    gpt2_extractor, test_loader, hook_manager, 
                    layer_vectors, alpha, judge_mlp, best_layer_idx
                )
                
                print(f"    Alpha {alpha}: F1={f1:.4f}, Acc={acc:.4f}, FPR={fpr:.4f}, TPR={tpr:.4f}")
                
                results.append({
                    'method': method,
                    'n_layers': n_layers,
                    'alpha': alpha,
                    'f1': f1,
                    'acc': acc,
                    'fpr': fpr,
                    'tpr': tpr
                })

    final_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(final_df)

    final_df.to_csv(os.path.join(SAVE_DIR, 'safety_alignment_results.csv'), index=False)# filepath: /home/anwesh/layers-to-latents/Part 2/toxic_vector_removal.py
import torch
import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from toxic_classification_gpt2 import GPT2Activations, ToxicCommentDataset, HarmfulDetectorMLP

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HarmfulVectorConstructor:
    def __init__(self, activation_dir):
        self.activation_dir = activation_dir

    def get_mean_difference_vector(self, layer_idx):
        data = np.load(os.path.join(self.activation_dir, f'train_layer_{layer_idx}_pooled_activations.npy'))
        labels = np.load(os.path.join(self.activation_dir, 'train_labels.npy'))
        
        toxic_mean = np.mean(data[labels==1], axis=0)
        non_toxic_mean = np.mean(data[labels==0], axis=0)
        vector = toxic_mean - non_toxic_mean
        return vector / np.linalg.norm(vector)

    def get_logistic_vector(self, layer_idx):
        data = np.load(os.path.join(self.activation_dir, f'train_layer_{layer_idx}_pooled_activations.npy'))
        labels = np.load(os.path.join(self.activation_dir, 'train_labels.npy'))
        
        clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        clf.fit(data, labels)
        
        vector = clf.coef_[0]
        return vector / np.linalg.norm(vector)

class SafetyHookManager:
    def __init__(self, model, device):
        self.model = model
        self.hooks = []
        self.device = device
    
    def register_hooks(self, layer_vectors, alpha):
        self.clear_hooks()
        
        for layer_idx, vector in layer_vectors.items():
            v = torch.tensor(vector, dtype=torch.float32, device=self.device)
            hook_fn = self._get_hook_fn(v, alpha)
            
            if 0 <= layer_idx < len(self.model.h):
                module = self.model.h[layer_idx]
                self.hooks.append(module.register_forward_hook(hook_fn))
                
    def _get_hook_fn(self, harmful_vector, alpha):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            v = harmful_vector.to(hidden_states.device).to(hidden_states.dtype)
            
            projections = torch.matmul(hidden_states, v)
            modified_hidden = hidden_states - alpha * projections.unsqueeze(-1) * v
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def evaluate_safety(gpt2_extractor, test_loader, hook_manager, layer_vectors, alpha, judge_mlp, judge_layer_idx):
    if alpha > 0:
        hook_manager.register_hooks(layer_vectors, alpha)
    else:
        hook_manager.clear_hooks()
    
    all_preds = []
    all_labels = []
    
    gpt2_extractor.model.eval()
    judge_mlp.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval alpha={alpha}", leave=False):
            input_ids = batch['input_ids'].to(gpt2_extractor.device)
            attention_mask = batch['attention_mask'].to(gpt2_extractor.device)
            labels = batch['label'].numpy()
            
            outputs = gpt2_extractor.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            judge_activations = outputs.hidden_states[judge_layer_idx + 1]
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(judge_activations.size()).float()
            sum_embeddings = torch.sum(judge_activations * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_activations = sum_embeddings / sum_mask
            
            mlp_out = judge_mlp(pooled_activations)
            preds = (mlp_out > 0.5).cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    hook_manager.clear_hooks()
    
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return f1, acc, fpr, tpr

def main():
    set_seed(42)
    
    SAVE_DIR = '/home/anwesh/scratch/layers-to-latents/gpt2_activations'
    RESULTS_CSV = os.path.join(SAVE_DIR, 'results.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        results_df = results_df.sort_values('test_f1', ascending=False)
        best_layer_idx = int(results_df.iloc[0]['layer_idx'])
        print(f"Best performing layer is: {best_layer_idx}")
    else:
        raise FileNotFoundError(f"Results file not found at {RESULTS_CSV}")

    torch.cuda.empty_cache()
    gc.collect()

    print("Initializing GPT-2 and DataLoader...")
    BATCH_SIZE = 32
    gpt2_extractor = GPT2Activations()
    test_dataset = ToxicCommentDataset(split='test', tokenizer=gpt2_extractor.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading Judge MLP for Layer {best_layer_idx}...")
    judge_mlp = HarmfulDetectorMLP(input_size=gpt2_extractor.h_size).to(DEVICE)
    judge_mlp.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'mlp_layer_{best_layer_idx}_model.pth')))

    alphas = [0.0, 0.5, 1.0]
    layer_counts = [1, 3, 5, 10]
    methods = ['mean_diff', 'logistic']

    results = []
    constructor = HarmfulVectorConstructor(SAVE_DIR)
    hook_manager = SafetyHookManager(gpt2_extractor.model, DEVICE)

    print("\nStarting Safety Alignment Experiments...")
    print("="*80)

    for method in methods:
        print(f"\nMethod: {method.upper()}")
        
        for n_layers in layer_counts:
            top_layers = results_df.head(n_layers)['layer_idx'].values.astype(int)
            print(f"  Editing Top {n_layers} Layers: {top_layers}")
            
            layer_vectors = {}
            for layer_idx in top_layers:
                if method == 'mean_diff':
                    vec = constructor.get_mean_difference_vector(layer_idx)
                else:
                    vec = constructor.get_logistic_vector(layer_idx)
                layer_vectors[layer_idx] = vec
                
            for alpha in alphas:
                torch.cuda.empty_cache()
                
                f1, acc, fpr, tpr = evaluate_safety(
                    gpt2_extractor, test_loader, hook_manager, 
                    layer_vectors, alpha, judge_mlp, best_layer_idx
                )
                
                print(f"    Alpha {alpha}: F1={f1:.4f}, Acc={acc:.4f}, FPR={fpr:.4f}, TPR={tpr:.4f}")
                
                results.append({
                    'method': method,
                    'n_layers': n_layers,
                    'alpha': alpha,
                    'f1': f1,
                    'acc': acc,
                    'fpr': fpr,
                    'tpr': tpr
                })

    final_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(final_df)

    final_df.to_csv(os.path.join(SAVE_DIR, 'safety_alignment_results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(SAVE_DIR, 'safety_alignment_results.csv')}")

if __name__ == '__main__':
    main()