import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from .model import SparseAutoencoder

class FeatureExplorer:
    def __init__(
        self,
        sae: SparseAutoencoder,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu"
    ):
        """
        Explores the features learned by a Sparse Autoencoder.
        Tracks top-activating contexts for each feature and analyzes arbitrary text.
        """
        self.sae = sae.eval().to(device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        self.d_sae = sae.d_sae
        self.d_model = sae.d_in
        
        # Hooks registry
        self.hook_handle = None
        self.layer_idx = None
        self.captured_acts = None
        
    def _register_hook(self, layer_idx: int):
        """Registers a hook to capture activations from the base model."""
        self.layer_idx = layer_idx
        self.captured_acts = None
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured_acts = output[0].detach()
            else:
                self.captured_acts = output.detach()
                
        target_layer = self.model.transformer.h[layer_idx]
        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        
    def _remove_hook(self):
        """Removes the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    @torch.no_grad()
    def build_feature_database(
        self,
        dataset,
        layer_idx: int,
        num_docs: int = 500,
        seq_len: int = 128,
        batch_size: int = 16,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Processes a subset of the dataset and records the top_k activating contexts
        for each feature in the SAE.
        """
        print(f"Building feature database using {num_docs} documents...")
        self._register_hook(layer_idx)
        
        # Initialize top-k activation tables
        # top_values: (d_sae, top_k)
        top_values = torch.zeros((self.d_sae, top_k), dtype=torch.float32, device=self.device)
        # Store documentation indices and token indices
        top_doc_indices = torch.full((self.d_sae, top_k), -1, dtype=torch.long, device=self.device)
        top_token_indices = torch.full((self.d_sae, top_k), -1, dtype=torch.long, device=self.device)
        
        # Store tokenized documents in memory to look up contexts later
        tokenized_docs = []
        raw_docs = []
        
        # Slice dataset
        docs_to_process = []
        for i in range(min(num_docs, len(dataset))):
            text = dataset[i].get("text", "")
            if text.strip():
                docs_to_process.append(text)
                raw_docs.append(text)
                
        # Batch process documents
        for batch_start in tqdm(range(0, len(docs_to_process), batch_size), desc="Extracting features"):
            batch_texts = docs_to_process[batch_start : batch_start + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len
            ).to(self.model.device)
            
            # Save the tokens
            for item in inputs["input_ids"].cpu().numpy():
                tokenized_docs.append(item)
                
            # Run forward pass to trigger hook
            self.model(**inputs)
            
            # captured_acts shape: (curr_batch_size, seq_len, d_model)
            acts = self.captured_acts.to(self.device)
            
            # Encode activations with SAE to get feature activations
            # Shape: (curr_batch_size * seq_len, d_sae)
            curr_batch_size = acts.shape[0]
            flat_acts = acts.view(-1, self.d_model)
            feature_acts = self.sae.encode(flat_acts)
            
            # Reshape feature activations back to (curr_batch_size, seq_len, d_sae)
            feature_acts = feature_acts.view(curr_batch_size, seq_len, self.d_sae)
            
            # Update top-k values for each feature
            # Iterate through batch items to map back to original documents
            for b_idx in range(curr_batch_size):
                doc_idx = batch_start + b_idx
                # acts_for_doc: (seq_len, d_sae)
                acts_for_doc = feature_acts[b_idx]
                
                # Check for active features (where activation > 0)
                # acts_for_doc transposed to (d_sae, seq_len)
                for feat_idx in range(self.d_sae):
                    feat_acts = acts_for_doc[:, feat_idx]
                    if feat_acts.max() == 0:
                        continue
                        
                    # Combine with existing top-k
                    combined_vals = torch.cat([top_values[feat_idx], feat_acts])
                    
                    # Create corresponding document and token index lists
                    new_doc_indices = torch.full((seq_len,), doc_idx, dtype=torch.long, device=self.device)
                    new_token_indices = torch.arange(seq_len, dtype=torch.long, device=self.device)
                    
                    combined_docs = torch.cat([top_doc_indices[feat_idx], new_doc_indices])
                    combined_tokens = torch.cat([top_token_indices[feat_idx], new_token_indices])
                    
                    # Sort and keep top-k
                    sorted_vals, sorted_indices = torch.sort(combined_vals, descending=True)
                    keep_indices = sorted_indices[:top_k]
                    
                    top_values[feat_idx] = sorted_vals[:top_k]
                    top_doc_indices[feat_idx] = combined_docs[keep_indices]
                    top_token_indices[feat_idx] = combined_tokens[keep_indices]
                    
        self._remove_hook()
        
        # Save results locally
        self.database = {
            "top_values": top_values.cpu().numpy(),
            "top_doc_indices": top_doc_indices.cpu().numpy(),
            "top_token_indices": top_token_indices.cpu().numpy(),
            "tokenized_docs": tokenized_docs,
            "raw_docs": raw_docs
        }
        
        return self.database

    def get_feature_summary(self, feature_idx: int, context_window: int = 5) -> List[Dict[str, Any]]:
        """
        Gets the top-activating contexts for a specific feature, formatting
        them with token highlights and activation values.
        """
        if not hasattr(self, "database"):
            raise ValueError("Feature database not built yet. Run build_feature_database() first.")
            
        vals = self.database["top_values"][feature_idx]
        docs = self.database["top_doc_indices"][feature_idx]
        tokens = self.database["top_token_indices"][feature_idx]
        tokenized_docs = self.database["tokenized_docs"]
        
        contexts = []
        for i in range(len(vals)):
            val = vals[i]
            if val <= 1e-5:
                # No more active tokens
                continue
                
            doc_idx = docs[i]
            token_idx = tokens[i]
            
            token_ids = tokenized_docs[doc_idx]
            
            # Extract window
            start_idx = max(0, token_idx - context_window)
            end_idx = min(len(token_ids), token_idx + context_window + 1)
            
            # Decode tokens in context window
            context_tokens = [self.tokenizer.decode([tid]) for tid in token_ids[start_idx:end_idx]]
            highlight_position = token_idx - start_idx
            
            contexts.append({
                "activation": float(val),
                "doc_idx": int(doc_idx),
                "token_idx": int(token_idx),
                "active_token": self.tokenizer.decode([token_ids[token_idx]]),
                "context_tokens": context_tokens,
                "highlight_position": highlight_position,
                "full_context": "".join(context_tokens)
            })
            
        return contexts

    @torch.no_grad()
    def analyze_custom_text(self, text: str, layer_idx: int) -> Dict[str, Any]:
        """
        Analyzes a custom text snippet. Runs it through the model and SAE,
        returning which features activated on each token and by how much.
        """
        self._register_hook(layer_idx)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]
        
        # Forward pass
        self.model(**inputs)
        self._remove_hook()
        
        # captured_acts: (1, seq_len, d_model)
        acts = self.captured_acts.to(self.device)
        seq_len = acts.shape[1]
        
        # Flat and encode
        flat_acts = acts.view(-1, self.d_model)
        feature_acts = self.sae.encode(flat_acts) # (seq_len, d_sae)
        
        # Find active features per token
        active_features_per_token = []
        for t_idx in range(seq_len):
            t_acts = feature_acts[t_idx]
            # Get indices and values of active features
            active_indices = torch.where(t_acts > 1e-5)[0].cpu().numpy()
            active_vals = t_acts[active_indices].cpu().numpy()
            
            # Sort by activation value descending
            sort_order = np.argsort(active_vals)[::-1]
            active_indices = active_indices[sort_order]
            active_vals = active_vals[sort_order]
            
            token_features = []
            for f_idx, val in zip(active_indices, active_vals):
                token_features.append({
                    "feature_idx": int(f_idx),
                    "activation": float(val)
                })
            
            active_features_per_token.append({
                "token": tokens[t_idx],
                "token_idx": t_idx,
                "features": token_features
            })
            
        return {
            "tokens": tokens,
            "analysis": active_features_per_token,
            # Also return summary of top-activating features overall in the sentence
            "top_features": self._get_sentence_top_features(feature_acts, tokens)
        }
        
    def _get_sentence_top_features(self, feature_acts: torch.Tensor, tokens: List[str]) -> List[Dict[str, Any]]:
        """Summarizes features with the highest activation in the sentence."""
        # Max activation for each feature across all tokens in the sentence
        max_acts, max_indices = torch.max(feature_acts, dim=0) # (d_sae,)
        
        # Get active features
        active_indices = torch.where(max_acts > 1e-5)[0].cpu().numpy()
        active_max_vals = max_acts[active_indices].cpu().numpy()
        active_token_indices = max_indices[active_indices].cpu().numpy()
        
        # Sort by max activation descending
        sort_order = np.argsort(active_max_vals)[::-1]
        active_indices = active_indices[sort_order]
        active_max_vals = active_max_vals[sort_order]
        active_token_indices = active_token_indices[sort_order]
        
        summary = []
        for f_idx, val, t_idx in zip(active_indices, active_max_vals, active_token_indices):
            summary.append({
                "feature_idx": int(f_idx),
                "max_activation": float(val),
                "activating_token": tokens[t_idx],
                "token_idx": int(t_idx)
            })
            
        return summary
