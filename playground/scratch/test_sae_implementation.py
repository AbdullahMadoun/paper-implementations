import torch
import sys

def test_sae():
    print("="*60)
    print("TESTING YOUR SPARSE AUTOENCODER IMPLEMENTATION")
    print("="*60)
    
    # 1. Try importing the model
    try:
        from src.model import SparseAutoencoder
    except ImportError as e:
        print("\033[91m[FAIL]\033[0m Could not import SparseAutoencoder. Ensure src/model.py exists.")
        print(f"Error details: {e}")
        return
    
    d_in = 768
    d_sae = 3072
    l1_coef = 3e-4
    
    # 2. Try initializing
    try:
        sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, l1_coef=l1_coef)
        print("\033[92m[PASS]\033[0m SparseAutoencoder class initialized successfully.")
    except Exception as e:
        print("\033[91m[FAIL]\033[0m SparseAutoencoder initialization threw an exception.")
        print(f"Error details: {e}")
        print("\nHint: Check if you have defined the __init__ method properly and initialized your parameters.")
        return
        
    # 3. Check parameters
    params = dict(sae.named_parameters())
    required_params = {
        "W_enc": (d_in, d_sae),
        "b_enc": (d_sae,),
        "W_dec": (d_sae, d_in),
        "b_dec": (d_in,)
    }
    
    params_ok = True
    for name, expected_shape in required_params.items():
        if name not in params:
            print(f"\033[91m[FAIL]\033[0m Missing parameter: '{name}'")
            params_ok = False
        else:
            actual_shape = tuple(params[name].shape)
            if actual_shape != expected_shape:
                print(f"\033[91m[FAIL]\033[0m Parameter '{name}' has shape {actual_shape}, expected {expected_shape}")
                params_ok = False
            else:
                print(f"\033[92m[PASS]\033[0m Parameter '{name}' exists with correct shape {expected_shape}.")
                
    if not params_ok:
        print("\nHint: Initialize these exact parameter names as nn.Parameter in your __init__.")
        return
        
    # 4. Check normalization function
    try:
        # Set decoder weights to random large values
        sae.W_dec.data = torch.randn(d_sae, d_in) * 10.0
        sae.make_decoder_weights_unit_norm()
        
        # Calculate norms of decoder rows
        norms = sae.W_dec.norm(p=2, dim=1)
        expected_norms = torch.ones(d_sae)
        
        if torch.allclose(norms, expected_norms, atol=1e-4):
            print("\033[92m[PASS]\033[0m make_decoder_weights_unit_norm() correctly forces unit norm on decoder rows.")
        else:
            print("\033[91m[FAIL]\033[0m make_decoder_weights_unit_norm() ran, but row norms were not 1.0.")
            print(f"Sample row norms: {norms[:5].tolist()}")
            params_ok = False
    except Exception as e:
        print("\033[91m[FAIL]\033[0m make_decoder_weights_unit_norm() threw an exception.")
        print(f"Error details: {e}")
        print("\nHint: You need to divide W_dec by its L2 norm along the correct dimension (rows of W_dec).")
        params_ok = False
        
    if not params_ok:
        return

    # 5. Check encode method
    x = torch.randn(8, d_in) # batch of 8
    try:
        feature_acts = sae.encode(x)
        if tuple(feature_acts.shape) != (8, d_sae):
            print(f"\033[91m[FAIL]\033[0m encode(x) returned shape {tuple(feature_acts.shape)}, expected (8, {d_sae})")
            return
        # Check ReLU active (no negative values)
        if (feature_acts < 0).any():
            print("\033[91m[FAIL]\033[0m encode(x) returned negative values! Did you forget ReLU?")
            return
        print("\033[92m[PASS]\033[0m encode(x) returned activations with correct shape and positive constraint.")
    except NotImplementedError:
        print("\033[93m[TODO]\033[0m encode(x) is not yet implemented.")
        return
    except Exception as e:
        print("\033[91m[FAIL]\033[0m encode(x) threw an exception.")
        print(f"Error details: {e}")
        return

    # 6. Check decode method
    try:
        x_hat = sae.decode(feature_acts)
        if tuple(x_hat.shape) != (8, d_in):
            print(f"\033[91m[FAIL]\033[0m decode(feature_acts) returned shape {tuple(x_hat.shape)}, expected (8, {d_in})")
            return
        print("\033[92m[PASS]\033[0m decode(feature_acts) reconstructed activations with correct shape.")
    except NotImplementedError:
        print("\033[93m[TODO]\033[0m decode(feature_acts) is not yet implemented.")
        return
    except Exception as e:
        print("\033[91m[FAIL]\033[0m decode(feature_acts) threw an exception.")
        print(f"Error details: {e}")
        return

    # 7. Check forward pass
    try:
        x_hat_f, feature_acts_f = sae(x)
        if tuple(x_hat_f.shape) != (8, d_in) or tuple(feature_acts_f.shape) != (8, d_sae):
            print("\033[91m[FAIL]\033[0m forward(x) returned incorrect shapes.")
            return
        print("\033[92m[PASS]\033[0m forward(x) pass successful.")
    except NotImplementedError:
        print("\033[93m[TODO]\033[0m forward(x) is not yet implemented.")
        return
    except Exception as e:
        print("\033[91m[FAIL]\033[0m forward(x) threw an exception.")
        print(f"Error details: {e}")
        return

    # 8. Check loss calculation
    try:
        loss, recon_loss, sparsity_loss = sae.compute_loss(x, x_hat, feature_acts)
        
        # Verify scalar outputs
        if loss.dim() != 0 or recon_loss.dim() != 0 or sparsity_loss.dim() != 0:
            print("\033[91m[FAIL]\033[0m compute_loss must return scalar tensors (dimension 0).")
            print(f"Shapes: loss={loss.shape}, recon={recon_loss.shape}, sparsity={sparsity_loss.shape}")
            return
            
        print("\033[92m[PASS]\033[0m compute_loss returned scalar loss terms.")
        
        # Verify weight loss balance
        # expected: loss = recon_loss + l1_coef * sparsity_loss
        expected_loss = recon_loss + l1_coef * sparsity_loss
        if not torch.allclose(loss, expected_loss, atol=1e-6):
            print(f"\033[91m[FAIL]\033[0m Total loss was {loss.item():.4f}, expected recon + l1_coef * sparsity ({expected_loss.item():.4f})")
            return
            
        print("\033[92m[PASS]\033[0m compute_loss formula verified.")
    except NotImplementedError:
        print("\033[93m[TODO]\033[0m compute_loss is not yet implemented.")
        return
    except Exception as e:
        print("\033[91m[FAIL]\033[0m compute_loss threw an exception.")
        print(f"Error details: {e}")
        return

    print("\n" + "="*60)
    print("\033[92mCONGRATULATIONS! ALL SAE MODEL TESTS PASSED!\033[0m")
    print("Your SparseAutoencoder model in src/model.py is fully correct.")
    print("Now proceed to implement SAETrainer.step_train in src/trainer.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_sae()
