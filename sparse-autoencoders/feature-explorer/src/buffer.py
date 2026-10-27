import torch
import random
from typing import Generator

class ActivationBuffer:
    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        layer_idx: int,
        buffer_size: int = 500_000,
        model_batch_size: int = 32,
        sae_batch_size: int = 4096,
        seq_len: int = 128,
        device: str = "cpu"
    ):
        """
        Shuffling buffer for language model activations.
        Extracts activations from `layer_idx` of the base `model` on the fly,
        shuffles them, and yields them in batches for training the SAE.
        
        Args:
            model: The base transformer language model.
            tokenizer: The tokenizer for the base model.
            dataset: The Hugging Face dataset.
            layer_idx: The block index of the transformer to hook (e.g., 6 for GPT-2).
            buffer_size: Maximum number of activation vectors to hold in memory.
            model_batch_size: Batch size of tokens to pass through the base LM.
            sae_batch_size: Batch size of activation vectors to yield for the SAE.
            seq_len: Sequence length for tokenization.
            device: PyTorch device ('cpu', 'cuda', 'mps').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.model_batch_size = model_batch_size
        self.sae_batch_size = sae_batch_size
        self.seq_len = seq_len
        self.device = device
        
        # We need the d_model size of the base model
        self.d_model = model.config.n_embd
        
        # Buffer initialization
        # Shape: (buffer_size, d_model)
        self.buffer = torch.zeros((buffer_size, self.d_model), dtype=torch.float32, device=device)
        self.pointer = 0
        self.num_filled = 0
        
        # Dataset pointer
        self.dataset_idx = 0
        self.dataset_size = len(dataset)
        
        # Hook target
        self.hook_handle = None
        self.temp_activations = []
        
    def _hook_fn(self, module, input, output):
        """Hook to capture activation outputs from a transformer layer."""
        # For Hugging Face models, block outputs are often tuples: (hidden_states, presents, attentions)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Keep on CPU or base device to save GPU memory until shuffling
        # We detach and clone to avoid memory leaks
        self.temp_activations.append(hidden_states.detach().to(self.device))

    def register_hook(self):
        """Registers the forward hook on the target layer."""
        target_layer = self.model.transformer.h[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        """Removes the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _fill_buffer(self):
        """Fills the activation buffer by running sequences through the base model."""
        self.temp_activations = []
        self.register_hook()
        
        # Keep track of how many vectors we need to fill the buffer
        target_vectors = self.buffer_size - self.pointer
        vectors_added = 0
        
        print(f"Refilling activation buffer (Target: {target_vectors} vectors)...")
        
        while vectors_added < target_vectors:
            # Prepare a batch of texts
            batch_texts = []
            while len(batch_texts) < self.model_batch_size:
                if self.dataset_idx >= self.dataset_size:
                    # Reset dataset index to loop if needed
                    self.dataset_idx = 0
                
                # Fetch text and handle potential differences in dataset structure
                item = self.dataset[self.dataset_idx]
                text = item.get("text", "")
                if text.strip():
                    batch_texts.append(text)
                self.dataset_idx += 1
            
            # Tokenize text with truncation and padding to fixed sequence length
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.seq_len
            ).to(self.model.device)
            
            # Run forward pass (no grads)
            with torch.no_grad():
                self.model(**inputs)
                
            # Pop the activations captured by the hook
            # Shape of each: (model_batch_size, seq_len, d_model)
            acts = torch.cat(self.temp_activations, dim=0)
            self.temp_activations = []
            
            # Flatten activations across batch and sequence dimensions
            # Shape: (model_batch_size * seq_len, d_model)
            acts_flat = acts.view(-1, self.d_model)
            
            # Determine how many vectors to copy
            num_to_copy = min(acts_flat.shape[0], target_vectors - vectors_added)
            
            # Copy into buffer
            start_idx = self.pointer
            end_idx = start_idx + num_to_copy
            
            self.buffer[start_idx:end_idx] = acts_flat[:num_to_copy].to(self.device)
            self.pointer += num_to_copy
            vectors_added += num_to_copy
            
        # Shuffle the buffer once it's full
        print(f"Buffer filled. Shuffling {self.buffer_size} vectors...")
        shuffled_indices = torch.randperm(self.buffer_size, device=self.device)
        self.buffer = self.buffer[shuffled_indices]
        
        # Reset pointer to start of buffer
        self.pointer = 0
        self.num_filled = self.buffer_size
        
        # Clean up hooks
        self.remove_hook()

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """Yields batches of shuffled activations."""
        while True:
            # If buffer is empty or near empty, fill it
            if self.num_filled - self.pointer < self.sae_batch_size:
                # Retain whatever residual we have left in the buffer by moving it to the front
                residual_size = self.num_filled - self.pointer
                if residual_size > 0:
                    self.buffer[0:residual_size] = self.buffer[self.pointer:self.num_filled]
                    self.pointer = residual_size
                else:
                    self.pointer = 0
                
                # Fill the rest of the buffer
                self._fill_buffer()
                
            # Yield next batch
            start = self.pointer
            end = start + self.sae_batch_size
            self.pointer = end
            yield self.buffer[start:end]
            
    def close(self):
        """Ensure hook is cleaned up."""
        self.remove_hook()
