import os
import json
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

from src.data import load_base_model, get_text_dataset
from src.pretrained import load_pretrained_sae
from src.explore import FeatureExplorer

app = FastAPI(title="Sparse Autoencoder Feature Explorer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
explorer = None
layer_idx = 6
device = "cpu"
db_path = "checkpoints/explorer_db.pt"

class TextAnalysisRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup():
    global explorer, device, layer_idx
    
    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Starting server. Using device: {device}")
    
    # Load base model & tokenizer
    model, tokenizer = load_base_model("gpt2", device=device)
    
    # Load pre-trained SAE (layer 6)
    try:
        sae = load_pretrained_sae(layer_idx=layer_idx, device=device)
    except NotImplementedError:
        print("\n" + "!"*80)
        print("WEB SERVER DELAYED:")
        print("SparseAutoencoder class in src/model.py is not yet implemented.")
        print("Please implement it first. Server will run but endpoints will return error codes.")
        print("!"*80 + "\n")
        sae = None
        
    explorer = FeatureExplorer(sae=sae, model=model, tokenizer=tokenizer, device=device)
    
    # Check if we have a cached database
    if sae is not None:
        if os.path.exists(db_path):
            print(f"Loading cached feature database from {db_path}...")
            try:
                explorer.database = torch.load(db_path, map_location="cpu")
                print("Database loaded successfully from cache.")
            except Exception as e:
                print(f"Failed to load cached database: {e}. Rebuilding...")
                build_and_cache_db(explorer)
        else:
            build_and_cache_db(explorer)

def build_and_cache_db(exp):
    dataset = get_text_dataset("NeelNanda/pile-10k")
    # Scan 150 documents (gives good features while loading fast - around 5-10s on CPU/MPS)
    exp.build_feature_database(
        dataset=dataset,
        layer_idx=layer_idx,
        num_docs=150,
        seq_len=128,
        batch_size=16,
        top_k=10
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    torch.save(exp.database, db_path)
    print(f"Saved feature database cache to {db_path}.")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "index.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    with open(dashboard_path, "r") as f:
        return f.read()

@app.get("/api/info")
def get_info():
    if explorer is None or explorer.sae is None:
        return {
            "status": "unimplemented",
            "message": "SparseAutoencoder model is not implemented in src/model.py yet."
        }
    return {
        "status": "ready",
        "d_in": explorer.sae.d_in,
        "d_sae": explorer.sae.d_sae,
        "layer_idx": layer_idx,
        "device": device,
        "database_docs": len(explorer.database["raw_docs"]) if hasattr(explorer, "database") else 0
    }

@app.get("/api/feature/{feat_idx}")
def get_feature(feat_idx: int):
    if explorer is None or explorer.sae is None:
        raise HTTPException(status_code=400, detail="SparseAutoencoder is not implemented yet.")
    if not hasattr(explorer, "database"):
        raise HTTPException(status_code=400, detail="Feature database is not loaded.")
    if feat_idx < 0 or feat_idx >= explorer.d_sae:
        raise HTTPException(status_code=404, detail=f"Feature index out of range (0-{explorer.d_sae - 1})")
        
    contexts = explorer.get_feature_summary(feat_idx, context_window=6)
    return {
        "feature_idx": feat_idx,
        "contexts": contexts
    }

@app.post("/api/analyze")
def analyze_text(req: TextAnalysisRequest):
    if explorer is None or explorer.sae is None:
        raise HTTPException(status_code=400, detail="SparseAutoencoder is not implemented yet.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        result = explorer.analyze_custom_text(req.text, layer_idx=layer_idx)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_features(query: str):
    """
    Search for features that activate highly on the queried token string (case insensitive).
    """
    if explorer is None or explorer.sae is None:
        raise HTTPException(status_code=400, detail="SparseAutoencoder is not implemented yet.")
    if not hasattr(explorer, "database"):
        raise HTTPException(status_code=400, detail="Feature database is not loaded.")
    if not query.strip():
        return {"query": query, "results": []}
        
    query_clean = query.strip().lower()
    matching_features = []
    
    # We inspect our database and look for features where the query string
    # appears in the top-k activating tokens.
    top_values = explorer.database["top_values"]
    top_doc_indices = explorer.database["top_doc_indices"]
    top_token_indices = explorer.database["top_token_indices"]
    tokenized_docs = explorer.database["tokenized_docs"]
    
    # Scan features
    for feat_idx in range(explorer.d_sae):
        feat_vals = top_values[feat_idx]
        feat_docs = top_doc_indices[feat_idx]
        feat_tokens = top_token_indices[feat_idx]
        
        # If the feature has active activations
        if feat_vals[0] > 1e-5:
            # Check top activating occurrences
            for i in range(len(feat_vals)):
                if feat_vals[i] <= 1e-5:
                    break
                
                doc_idx = feat_docs[i]
                tok_idx = feat_tokens[i]
                
                # Retrieve token string
                token_id = tokenized_docs[doc_idx][tok_idx]
                token_str = explorer.tokenizer.decode([token_id]).strip().lower()
                
                if query_clean in token_str or token_str in query_clean:
                    matching_features.append({
                        "feature_idx": feat_idx,
                        "activating_token": explorer.tokenizer.decode([token_id]),
                        "max_activation": float(feat_vals[0]),
                        "example_activation": float(feat_vals[i])
                    })
                    break # Only add once per feature
                    
    # Sort results by max activation descending
    matching_features = sorted(matching_features, key=lambda x: x["max_activation"], reverse=True)
    
    return {
        "query": query,
        "results": matching_features[:50] # limit to 50 results
    }

def run_server(host="127.0.0.1", port=8000):
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run("src.server:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    run_server()
