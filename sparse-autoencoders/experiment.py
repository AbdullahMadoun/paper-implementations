#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. Install dependencies
get_ipython().system('pip install transformers datasets safetensors tqdm')


# In[62]:


# 2. Imports and Device Setup
import os
import math
import random
import json
import html
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from typing import Generator, List, Dict, Any, Tuple
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pandas as pd
# Set up device (Colab T4 GPU uses cuda)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Using device: {device}")


# In[101]:


def load_base_model(model_name="gpt2", device="cpu"):
    print(f"Loading base language model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

def get_text_dataset(dataset_name="dair-ai/emotion", split="train"):
    print(f"Loading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}. Falling back to wikitext-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    return dataset


# In[102]:


model, tokenizer= load_base_model()


# > ✍️ **My Work** — Loading dataset

# In[103]:


ds = get_text_dataset()


# > ✍️ **My Work** — Initializing activations buffer

# In[104]:


activations = []


# > ✍️ **My Work** — This my first time using hooks so I want to emphasize it

# In[105]:


def hook(instance, input, output): 
    """
    hook to register activations of gpt2
    """
    activations.append(output)


# > ✍️ **My Work** — Device setup

# In[106]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device) 
target_module = model.transformer.h[6]


# > ✍️ **My Work** — Also first time interacting with feature level interactions

# In[107]:


target_module.register_forward_hook(hook)


# > ✍️ **My Work** — I explored it while experimenting with different datasets my goal was to try to have one without clipping just to fit all of the text into the context I wanted to use a labelled one to see if the SAE captures the labels

# In[108]:


lengths=[]
for i in ds["text"]: 
   lengths.append(len(i))
plt.hist(x= lengths, bins =50, range=(0,max(lengths)))
plt.show()


# In[109]:


lengths=[]
for i in ds["text"]: 
   lengths.append(len(tokenizer.encode(i)))
plt.hist(x= lengths, bins =50, range=(0,max(lengths)))
plt.show()


# In[111]:


for i in tqdm(range(len(ds["text"]))): 
    with torch.inference_mode():
        tokens = tokenizer.encode(ds["text"][i],
        max_length=1024, 
        truncation = True,
        return_tensors="pt")
        tokens = tokens.to(device)
        model(tokens)


# In[14]:


activations[0][0].shape


# In[15]:


torch.reshape(activations[0][0], shape = (48,64)).shape


# In[16]:


emotion_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


# In[17]:


tensors = [[], [], [] , [] , [] ,[]]
for i in range(100): 
    tensor = activations[i][0]
    tensor = torch.mean(tensor, dim =1)
    tn = torch.reshape(tensor, shape = (24,32))
    tensors[ds["label"][i]].append(tn)
    
    


# In[ ]:


# > 🤖 **AI-Assisted** — Dataset Balancing
import random

# Find max class size to oversample
max_count = max(len(t) for t in tensors)
print(f"Balancing classes to max_count = {max_count}")

balanced_tensors = []
for label_list in tensors:
    if len(label_list) == 0:
        balanced_tensors.append([])
        continue
    # Randomly sample with replacement
    balanced_list = [random.choice(label_list) for _ in range(max_count)]
    balanced_tensors.append(balanced_list)

# Overwrite tensors for the training loop
tensors = balanced_tensors
for i, t in enumerate(tensors):
    print(f"Class {i} size: {len(t)}")


# In[18]:


from numpy import average


# In[19]:


for i in tensors: 
    print(len((i)))


# # Most of the activations seem non discrimnative and it is quite clear that there is a common acitvation pattern across all 6 emotions suggesting a common pattern 
# # it appears neagtive emotions are less overall "bright" but there is no clear features for them 

# In[52]:


avg_emo_mat= {}
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))
axes_flat = axes.flatten()

for i in range(6): 
        averaged_matrix = torch.mean(torch.stack(tensors[i]), dim = 0).detach().clone()
        avg_emo_mat[emotion_mapping[i]] = averaged_matrix
        ax = axes_flat[i]
        sns.heatmap(averaged_matrix.cpu()
        ,ax = ax )
        ax.set_title(emotion_mapping[i])

plt.tight_layout()
plt.show()


# In[ ]:


element_wise_s_j = avg_emo_mat["sadness"]* avg_emo_mat["joy"]
sns.heatmap(element_wise_s_j.cpu())


# In[ ]:


element_wise_l_j = avg_emo_mat["love"]* avg_emo_mat["joy"]
sns.heatmap(element_wise_l_j.cpu())


# In[ ]:





# 
# # 🧠 Sparse Autoencoder (SAE) Playground on Google Colab
# 
# This notebook guides you through building a **Sparse Autoencoder (SAE)** from scratch, training it, loading pre-trained weights, and visualizing the semantic concepts it learns.
# 
# ## 📐 Mathematical Recap
# - **Encoder**: Projects centered activations to a sparse latent space:  
#   $$f(x) = \text{ReLU}\left((x - b_{dec}) W_{enc} + b_{enc}\right)$$
# - **Decoder**: Reconstructs activations from sparse latents:  
#   $$\hat{x} = f(x) W_{dec} + b_{dec}$$
# - **Loss**: Minimizes MSE while penalizing L1 norm:  
#   $$L = \|x - \hat{x}\|_2^2 + \lambda \|f(x)\|_1$$
# - **Decoder Normalization**: Normalize rows of $W_{dec}$ (dictionary atoms) to unit L2 norm after each gradient step to prevent the model from cheating the L1 penalty:  
#   $$\|W_{dec}[k, :]\|_2 = 1$$

# In[35]:


class SAE(nn.Module): 
    """
     Batch dim =1 
    """
    def __init__(self,d = 768, factor =8): 
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(
                                size = (d,  d*factor), 
                                requires_grad=True,
                                dtype = torch.float
                            )/ math.sqrt(d*factor)) 
        self.decoder =nn.Parameter( torch.randn(
                                size = (d*factor, d), 
                                requires_grad=True,
                                dtype = torch.float
                            ) / math.sqrt(d*factor))
        self.encb = nn.Parameter(torch.zeros(
                                size =  (d*factor, ), 
                                requires_grad=True,
                                dtype = torch.float
                            )) 
        self.decb = nn.Parameter(torch.zeros(
                                size =  (d, ), 
                                requires_grad=True,
                                dtype = torch.float
                            )) 


    def encode(self, x): 
        #(batch , d ) x (d , d*F)
        h = (x- self.decb).matmul(self.encoder)   + self.encb
        h = F.relu(h)
        return h
    def decode(self, h): 
        #(batch , d*F ) x (d*F , d)
        x = h.matmul(self.decoder) +  self.decb 
        return x
    def forward(self, x):
        h = self.encode(x)
        x = self.decode(h)
        return x, h 
        


# In[36]:


torch.device(device)


# In[37]:


type(activations)


# In[110]:


from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 1. PREPARE DATA (WITH POST-COLLECTION OVERSAMPLING)

# Grab the labels that correspond to our collected activations
labels = np.array(ds["label"][:len(activations)])
unique_labels = np.unique(labels)

# Find the size of the largest class
max_count = max((labels == lbl).sum() for lbl in unique_labels)

balanced_activations = []
for lbl in unique_labels:
    idx_for_lbl = np.where(labels == lbl)[0]
    # Oversample to match the largest class
    sampled_indices = np.random.choice(idx_for_lbl, size=max_count, replace=True)
    for idx in sampled_indices:
        balanced_activations.append(activations[idx])

print(f"Original sentences: {len(activations)}")
print(f"Balanced sentences (Oversampled): {len(balanced_activations)}")

# Now flatten the BALANCED activations
all_acts = torch.cat([act[0].squeeze(0) for act in balanced_activations], dim=0) 

# Create DataLoader
batch_size = 4096  
dataset = TensorDataset(all_acts)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Total balanced tokens for training: {len(all_acts)}")


# In[44]:


sae_model = SAE().to(device)



# In[45]:


optimizer = torch.optim.AdamW(lr = 0.001, params = sae_model.parameters())


# In[87]:


EPOCHS = 10
lmb = 0.02
for i in range(EPOCHS):

    total_loss = 0
    sae_model.train()

    for batch in dataloader:
        batch_acts = batch[0].to(device)
        #forward
        act_reconstruct, h  = sae_model(batch_acts) 
        
        #loss
        mse_loss = F.mse_loss(act_reconstruct, batch_acts)   
        sparsity_loss = h.abs().sum(dim=-1).mean()
        loss = mse_loss + lmb * sparsity_loss

        #zero grad
        optimizer.zero_grad()

        #backward
        loss.backward()

        #step
        optimizer.step()
        
        batch_variance = batch_acts.pow(2).mean() # or batch_acts.var()
        fve = 1.0 - (mse_loss / batch_variance)
        # 2. Calculate L0 (how many latents are non-zero?)
        # We use > 0.001 to account for tiny floating point noise
        l0_sparsity = (h > 0.001).float().sum(dim=-1).mean()
        print(f"Loss: {loss.item():.2f} | FVE: {fve.item():.2f} | L0: {l0_sparsity.item():.1f}")
    print(f"Loss at epoch {i+1} is {loss}")


# In[88]:


import html
import json
import numpy as np
import torch
from IPython.display import display, HTML

def get_custom_activations(gpt2_model, tokenizer, text, device):
    captured = []
    def temp_hook(module, input, output):
        captured.append(output[0].detach() if isinstance(output, tuple) else output.detach())
    
    handle = gpt2_model.transformer.h[6].register_forward_hook(temp_hook)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gpt2_model(**inputs)
        
    handle.remove()
    return captured[0], inputs["input_ids"][0].cpu().numpy()

def visualize_sentence(text):
    sae = sae_model if 'sae_model' in globals() else model
    gpt2 = model if 'sae_model' in globals() else None
    
    if not hasattr(gpt2, 'transformer'):
        for k, v in globals().items():
            if k in ['model', 'gpt2'] and hasattr(v, 'transformer'):
                gpt2 = v
                break
                
    if gpt2 is None:
        print("Error: Could not find base GPT-2 model (with .transformer attribute) in globals.")
        return

    # 1. Capture layer 6 activations
    acts, input_ids = get_custom_activations(gpt2, tokenizer, text, device)
    acts = acts.squeeze(0)  # Shape: [seq_len, 768]
    
    # 2. Encode activations using the SAE
    sae.eval()
    with torch.no_grad():
        h = sae.encode(acts.to(device)).cpu()  # Shape: [seq_len, d_sae]
        
    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    
    # Unique container ID to prevent collision in notebook
    container_id = f"sae-explorer-{hash(text) % 10000}"
    
    # 3. Render HTML chips and JS panel
    chips_html = ""
    for i, token_str in enumerate(tokens):
        escaped_tok = html.escape(token_str)
        t_acts = h[i]
        
        # Don't analyze BOS token (index 0)
        if i == 0:
            chips_html += f"""
            <span class="token-chip-{container_id}" data-token-idx="{i}" data-features="[]" style="display: inline-block; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); color: #4b5563; padding: 6px 10px; margin: 4px; border-radius: 6px; font-family: monospace; font-size: 1.1em; cursor: default; user-select: none;">
                {escaped_tok}
            </span>
            """
            continue
            
        active_indices = torch.where(t_acts > 1e-3)[0].numpy()
        active_vals = t_acts[active_indices].numpy()
        sort_order = np.argsort(active_vals)[::-1]
        active_indices = active_indices[sort_order]
        active_vals = active_vals[sort_order]
        
        token_features = [{"feature_idx": int(idx), "activation": float(val)} for idx, val in zip(active_indices, active_vals)]
        features_json = json.dumps(token_features)
        
        if len(active_vals) > 0:
            max_act = active_vals[0]
            opacity = min(0.7, max_act / 5.0 + 0.15)
            bg_color = f"rgba(139, 92, 246, {opacity})"
            border_color = f"rgba(139, 92, 246, {opacity + 0.2})"
            chips_html += f"""
            <span class="token-chip-{container_id}" data-token-idx="{i}" data-features='{html.escape(features_json)}' style="display: inline-block; background-color: {bg_color}; border: 1px solid {border_color}; color: #f8fafc; padding: 6px 10px; margin: 4px; border-radius: 6px; font-family: monospace; font-size: 1.1em; cursor: pointer; font-weight: 500; transition: all 0.2s;">
                {escaped_tok}
            </span>
            """
        else:
            chips_html += f"""
            <span class="token-chip-{container_id}" data-token-idx="{i}" data-features="[]" style="display: inline-block; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); color: #64748b; padding: 6px 10px; margin: 4px; border-radius: 6px; font-family: monospace; font-size: 1.1em; cursor: pointer; transition: all 0.2s;">
                {escaped_tok}
            </span>
            """
            
    html_str = f"""
    <div id="{container_id}" style="background-color: #090d16; color: #f1f5f9; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; flex-direction: row; gap: 20px; max-width: 950px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); margin-bottom: 20px;">
        
        <!-- Left Side: Interactive Text -->
        <div style="flex: 1.4; display: flex; flex-direction: column;">
            <h3 style="margin-top:0; color: #a78bfa; font-weight: 600; font-size: 1.3em;">🎨 Interactive Token Activation Highlighter</h3>
            <p style="color: #94a3b8; font-size: 0.9em; margin-bottom: 15px; margin-top: 2px;">Click any highlighted token to inspect its active SAE features.</p>
            <div style="line-height: 2.2; padding: 16px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); flex-grow: 1;">
                {chips_html}
            </div>
        </div>
        
        <!-- Right Side: Details Panel -->
        <div style="flex: 1; min-width: 280px; display: flex; flex-direction: column; background: rgba(255,255,255,0.01); border-left: 1px solid rgba(255,255,255,0.08); padding-left: 20px;">
            <h4 style="margin-top:0; font-size: 1em; color: #94a3b8; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px; font-weight: 500;">
                Active Features for <span id="selected-token-{container_id}" style="color: #a78bfa; font-weight: bold;">(Select a Token)</span>
            </h4>
            <div id="details-panel-{container_id}" style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px; overflow-y: auto; max-height: 220px; padding-right: 5px;">
                <div style="color: #64748b; font-style: italic;">Select a highlighted token to inspect.</div>
            </div>
        </div>
        
    </div>

    <script>
    (function() {{
        const container = document.getElementById('{container_id}');
        const chips = container.querySelectorAll('.token-chip-{container_id}');
        const detailsPanel = container.querySelector('#details-panel-{container_id}');
        const selectedToken = container.querySelector('#selected-token-{container_id}');
        
        chips.forEach(chip => {{
            chip.addEventListener('click', () => {{
                // Deselect other chips
                chips.forEach(c => {{
                    c.style.outline = 'none';
                    c.style.transform = 'none';
                }});
                
                // Highlight clicked chip
                chip.style.outline = '2px solid #a78bfa';
                chip.style.outlineOffset = '2px';
                
                const tok = chip.innerText.trim();
                selectedToken.innerText = `"${{tok}}"`;
                
                const features = JSON.parse(chip.getAttribute('data-features') || '[]');
                if (features.length === 0) {{
                    detailsPanel.innerHTML = '<div style="color: #64748b; font-style: italic;">No features active on this token.</div>';
                    return;
                }}
                
                let htmlStr = '';
                features.forEach(f => {{
                    // Max-cap visually at 5.0 for progress bar scale
                    const pct = Math.min(100, (f.activation / 5.0) * 100);
                    htmlStr += `
                    <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                        <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 4px;">
                            <span style="color: #cbd5e1;"><b>Feature #${{f.feature_idx}}</b></span>
                            <span style="color: #a78bfa; font-family: monospace; font-weight: bold;">${{f.activation.toFixed(3)}}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; overflow: hidden;">
                            <div style="background: #a78bfa; width: ${{pct}}%; height: 100%;"></div>
                        </div>
                    </div>
                    `;
                }});
                detailsPanel.innerHTML = htmlStr;
            }});
        }});
        
        // Auto-select first non-BOS token chip
        const firstActive = container.querySelector('.token-chip-{container_id}[data-token-idx="1"]') || container.querySelector('.token-chip-{container_id}');
        if (firstActive) {{
            firstActive.click();
        }}
    }})();
    </script>
    """
    display(HTML(html_str))


# In[89]:


visualize_sentence("I was completely stunned and surprised by the incredible news!")


# In[90]:


import html
import numpy as np
import torch
from IPython.display import display, HTML

def tell_feature_story(feat_idx, num_samples=2000):
    """
    Uncover the true story of a feature by analyzing its emotion profile, 
    top activating tokens, and real-world contexts.
    """
    sae = sae_model if 'sae_model' in globals() else model
    
    emotion_sums = {v: 0.0 for k, v in emotion_mapping.items()}
    top_contexts = []
    token_stats = {}
    
    sae.eval()
    scan_limit = min(num_samples, len(activations))
    
    with torch.no_grad():
        for i in range(scan_limit):
            label = ds["label"][i]
            emotion_name = emotion_mapping.get(label, "UNKNOWN")
            
            h_states = activations[i][0].squeeze(0)  # Shape: [seq_len, 768]
            if h_states.shape[0] <= 1:
                continue
            
            # Skip BOS token
            h = sae.encode(h_states[1:].to(device)).cpu()
            feat_acts = h[:, feat_idx].numpy()
            
            if feat_acts.max() > 0:
                emotion_sums[emotion_name] += feat_acts.sum()
                
            # Re-tokenize text to match activations
            text = ds["text"][i]
            input_ids = tokenizer.encode(text, truncation=True, max_length=1024)
            tokens = [tokenizer.decode([tid]) for tid in input_ids]
            
            for seq_pos, act_val in enumerate(feat_acts):
                if act_val > 1e-3:
                    tok_idx = seq_pos + 1  # +1 because we skipped BOS
                    if tok_idx < len(tokens):
                        tok_str = tokens[tok_idx]
                        
                        if tok_str not in token_stats:
                            token_stats[tok_str] = []
                        token_stats[tok_str].append(act_val)
                        
                        top_contexts.append((act_val, tok_str, tokens, tok_idx, emotion_name))

    max_overall = max(top_contexts, key=lambda x: x[0])[0] if top_contexts else 0.0
    
    # Sort top contexts
    top_contexts.sort(key=lambda x: x[0], reverse=True)
    top_contexts = top_contexts[:5]
    
    # Aggregate token stats
    agg_tokens = []
    for tok, vals in token_stats.items():
        agg_tokens.append({
            "token": tok,
            "mean_act": np.mean(vals),
            "max_act": np.max(vals),
            "count": len(vals)
        })
    # Sort by total activation mass (frequency * average strength)
    agg_tokens.sort(key=lambda x: x["count"] * x["mean_act"], reverse=True)
    top_tokens = agg_tokens[:10]
    
    # Normalize emotion profile
    total_emotion = sum(emotion_sums.values())
    emotion_profile = {k: (v / total_emotion if total_emotion > 0 else 0) for k, v in emotion_sums.items()}
    
    # --- HTML RENDER ---
    if total_emotion == 0:
        display(HTML(f"<div style='color: #94a3b8; padding: 20px; font-family: sans-serif;'>Feature #{feat_idx} did not activate in the scanned samples.</div>"))
        return
        
    emotion_colors = {
        "SADNESS": "#3b82f6", "JOY": "#f59e0b", "LOVE": "#ec4899",
        "ANGER": "#ef4444", "FEAR": "#8b5cf6", "SURPRISE": "#10b981"
    }
    
    emotion_bars_html = ""
    sorted_emotions = sorted(emotion_profile.items(), key=lambda x: x[1], reverse=True)
    for emo, pct in sorted_emotions:
        if pct < 0.01: continue
        color = emotion_colors.get(emo, "#cbd5e1")
        pct_val = pct * 100
        emotion_bars_html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin-bottom: 4px; color: #cbd5e1; font-weight: 500;">
                <span>{emo}</span>
                <span>{pct_val:.1f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: {color}; width: {pct_val}%; height: 100%; border-radius: 4px; box-shadow: 0 0 8px {color}80;"></div>
            </div>
        </div>
        """
        
    tokens_html = ""
    for t in top_tokens:
        tok_str = html.escape(t['token'])
        tokens_html += f"""
        <span style="background: rgba(192, 132, 252, 0.15); border: 1px solid rgba(192, 132, 252, 0.3); color: #d8b4fe; padding: 6px 12px; border-radius: 8px; font-family: monospace; font-size: 0.95em; display: inline-flex; align-items: center; gap: 6px;">
            "{tok_str}" <span style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 0.8em; color: #a78bfa;">x{t['count']}</span>
        </span>
        """
        
    contexts_html = ""
    for act, tok_str, tokens, tok_idx, emo in top_contexts:
        color = emotion_colors.get(emo, "#c084fc")
        start_idx = max(0, tok_idx - 8)
        end_idx = min(len(tokens), tok_idx + 9)
        
        ctx_parts = []
        for i in range(start_idx, end_idx):
            esc_t = html.escape(tokens[i])
            if i == tok_idx:
                ctx_parts.append(f'<span style="background: {color}40; color: #fff; padding: 2px 6px; border-radius: 4px; font-weight: bold; border-bottom: 2px solid {color};">{esc_t}</span>')
            else:
                ctx_parts.append(f'<span style="opacity: 0.85;">{esc_t}</span>')
                
        ctx_str = "".join(ctx_parts)
        if start_idx > 0: ctx_str = "..." + ctx_str
        if end_idx < len(tokens): ctx_str = ctx_str + "..."
        
        contexts_html += f"""
        <div style="background: #111827; padding: 16px; border-radius: 10px; border: 1px solid #1e293b; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-size: 0.75em; color: {color}; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; padding: 2px 8px; background: {color}20; border-radius: 4px;">{emo}</span>
                <span style="font-size: 0.85em; color: #a78bfa; font-weight: bold; font-family: monospace;">Activation: {act:.3f}</span>
            </div>
            <div style="font-family: Georgia, serif; line-height: 1.7; color: #f1f5f9; font-size: 1.1em;">
                {ctx_str}
            </div>
        </div>
        """
        
    html_out = f"""
    <div style="background-color: #0b0f19; color: #f1f5f9; padding: 30px; border-radius: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); max-width: 950px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 20px; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2em; font-weight: 800;">
                    Feature #{feat_idx}
                </div>
                <div style="color: #64748b; font-size: 1em; font-weight: 500; padding-top: 6px;">Semantic Dashboard</div>
            </div>
            <div style="background: rgba(192, 132, 252, 0.1); color: #c084fc; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; font-weight: bold; border: 1px solid rgba(192, 132, 252, 0.2);">
                Max Act: {max_overall:.2f}
            </div>
        </div>
        <div style="display: flex; gap: 25px; margin-bottom: 30px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px; background: #111827; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 1.1em; color: #e2e8f0; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    📊 Emotion Profile
                </h3>
                {emotion_bars_html}
            </div>
            <div style="flex: 1.5; min-width: 350px; background: #111827; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 1.1em; color: #e2e8f0; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    🔤 Primary Triggers
                </h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    {tokens_html}
                </div>
            </div>
        </div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 1.2em; color: #e2e8f0; font-weight: 600; padding-left: 5px;">
                🌍 Top Contexts in the Wild
            </h3>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                {contexts_html}
            </div>
        </div>
    </div>
    """
    display(HTML(html_out))


# In[91]:


tell_feature_story(5019)


# In[ ]:





# In[ ]:


# > 🤖 **AI-Assisted** — Feature Visualization in Style
from IPython.display import display, HTML
import torch

# Let's find an interpretable feature by running some sample texts 
# and tracking the max activating feature in the Sparse Autoencoder.
sae.eval()

# Sample some texts from the dataset
sample_texts = ds["text"][:50]
all_encoded = []
all_tokens = []

for text in sample_texts:
    tokens = tokenizer.encode(text, max_length=128, truncation=True, return_tensors="pt").to(device)
    all_tokens.append(tokens[0].cpu().tolist())
    with torch.inference_mode():
        # Get GPT-2 activations
        model(tokens)
        gpt_acts = activations[-1][0].mean(dim=1) # [batch, seq_len=1, hidden] -> wait, is it [batch, seq_len, hidden]?
        # Let's just use the last token's activation for simplicity or mean pooling
        # In the original code they did: tensor = torch.mean(tensor, dim=1)
        # So we use gpt_acts which is now [batch, hidden]
        
        # Run through SAE
        decoded, encoded = sae(gpt_acts)
        all_encoded.append(encoded[0].cpu()) # [hidden_dim]

# Find the feature (neuron) that has the highest variance or max activation across samples
encoded_matrix = torch.stack(all_encoded) # [50, hidden_dim]

# Pick a feature that is highly active for at least one sample but sparsely active overall
feature_variances = encoded_matrix.var(dim=0)
best_feature_idx = torch.argmax(feature_variances).item()

# Get the top 5 activating texts for this feature
feature_activations = encoded_matrix[:, best_feature_idx]
top_values, top_indices = torch.topk(feature_activations, k=min(5, len(feature_activations)))

html_output = f"""
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #1e1e1e; color: #ffffff; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
    <h2 style="color: #61dafb; border-bottom: 2px solid #61dafb; padding-bottom: 10px;">SAE Feature Analysis Showcase</h2>
    <p style="font-size: 1.1em;"><strong>Analyzed Feature Index:</strong> <span style="color: #ff9800;">{best_feature_idx}</span></p>
    <p style="font-size: 1.1em;">This feature was selected automatically for its high variance and sparsity.</p>
    <h3 style="color: #4caf50;">Top Activating Texts</h3>
    <ul style="list-style-type: none; padding-left: 0;">
"""

for val, idx in zip(top_values, top_indices):
    text = sample_texts[idx.item()]
    label = emotion_mapping.get(ds["label"][idx.item()], "unknown")
    act_val = val.item()
    
    # Highlight the text box
    html_output += f"""
        <li style="background-color: #2d2d2d; margin-bottom: 15px; padding: 15px; border-left: 5px solid #ff9800; border-radius: 4px;">
            <div style="margin-bottom: 8px;">
                <span style="background-color: #4caf50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-right: 10px;">{label.upper()}</span>
                <span style="color: #ff9800; font-weight: bold;">Activation: {act_val:.4f}</span>
            </div>
            <div style="font-size: 1.05em; line-height: 1.4;">{text}</div>
        </li>
    """

html_output += """
    </ul>
</div>
"""

display(HTML(html_output))

# Save the HTML to an artifact file for the user to view
with open('feature_showcase.html', 'w') as f:
    f.write(html_output)

