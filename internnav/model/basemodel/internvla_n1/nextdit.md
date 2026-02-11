# NextDiT Architecture Explanation

This document details the architecture of NextDiT, a transformer-based model for conditional generation, likely used in a diffusion process for tasks like trajectory prediction. The architecture is primarily defined by three components across two files: `LuminaNextDiTBlock`, `LuminaNextDiT2DModel` (in `nextdit_traj.py`), and the main wrapper `NextDiTCrossAttn` (in `nextdit_crossattn_traj.py`).

The overall architecture is that of a **DiT (Diffusion Transformer)**, which uses a transformer backbone to denoise a latent representation, conditioned by external inputs via cross-attention.

---

## 1. High-Level Overview

The model consists of three main classes:

1.  **`NextDiTCrossAttn`**: The top-level interface. It wraps the core transformer model, preparing inputs and passing them to the main processing unit.
2.  **`LuminaNextDiT2DModel`**: The core transformer model. It handles the initial embedding of inputs, the sequential processing through transformer layers, and the final output projection.
3.  **`LuminaNextDiTBlock`**: The fundamental building block of the transformer. Each block performs self-attention on the input sequence, cross-attention with a conditioning sequence, and processing through a feed-forward network.

---

## 2. Module Details and Data Flow

### 2.1. `NextDiTCrossAttn` (Wrapper)

This is the main entry point for the model.

-   **Purpose**: To receive the primary inputs (`x`, `timestep`, `z_latents`) and orchestrate the denoising prediction by calling the underlying `LuminaNextDiT2DModel`.
-   **Inputs**:
    -   `x` (`torch.Tensor`): The noisy input latent tensor. Shape: `(batch_size, in_channels, height, width)`.
    -   `timestep` (`torch.Tensor`): The current diffusion timestep for each item in the batch. Shape: `(batch_size,)`.
    -   `z_latents` (`torch.Tensor`): The conditioning latent tensor (e.g., embeddings from a text encoder, goal information, etc.). Shape: `(batch_size, seq_len, latent_embedding_size)`.
-   **Processing**:
    1.  It takes the inputs `x`, `timestep`, and `z_latents`.
    2.  It creates a non-restrictive attention mask for `z_latents`, indicating that all parts of the conditioning latent are visible.
    3.  It calls `self.model` (an instance of `LuminaNextDiT2DModel`), passing the inputs.
-   **Output**:
    -   `model_pred` (`torch.Tensor`): The predicted denoised latent from the core model. Shape is the same as the input `x`.

### 2.2. `LuminaNextDiT2DModel` (Core Transformer)

This class implements the main transformer structure.

-   **Purpose**: To convert the input latent image into a sequence of patches, process them through a series of transformer blocks while injecting conditioning information, and project the result back to the latent image space.
-   **Inputs**:
    -   `hidden_states` (`torch.Tensor`): The noisy input latent (from `x` in the wrapper).
    -   `timestep` (`torch.Tensor`): The diffusion timestep.
    -   `encoder_hidden_states` (`torch.Tensor`): The conditioning latents (from `z_latents` in the wrapper).
    -   `encoder_mask` (`torch.Tensor`): Attention mask for the conditioning latents.
    -   `image_rotary_emb` (`torch.Tensor`): Pre-computed rotary position embeddings to provide spatial information to the model.
-   **Internal Modules**:
    -   `patch_embedder`: Converts the 2D latent input into a 1D sequence of embedded patches.
    -   `caption_projection`: A linear layer to project the `encoder_hidden_states` to the model's internal dimension (`hidden_size`).
    -   `time_caption_embed`: Combines the `timestep` embedding and the (summed) `encoder_hidden_states` to create a single conditioning vector `temb`. This vector is used for adaptive normalization within the transformer blocks.
    -   `layers`: A `ModuleList` containing multiple instances of `LuminaNextDiTBlock`.
    -   `norm_out`: A final adaptive normalization layer that also projects the processed patch sequence back into the shape required for the output latent image.
-   **Processing**:
    1.  The `encoder_hidden_states` (conditioning) are projected to the model's dimension.
    2.  A combined time and conditioning embedding (`temb`) is created.
    3.  The input `hidden_states` (latent image) are converted into a sequence of patch embeddings (although this step appears to be missing from the provided `forward` method, it is a standard part of DiT architectures and implied by the `patch_embedder` module).
    4.  The patch embeddings are processed sequentially through each `LuminaNextDiTBlock` in `self.layers`.
    5.  The output from the final block is processed by `norm_out`, which applies adaptive normalization using `temb` and reshapes the sequence back into a latent image.
-   **Output**:
    -   `sample` (`torch.Tensor`): The final predicted denoised latent. Shape: `(batch_size, out_channels, height, width)`.

### 2.3. `LuminaNextDiTBlock` (Fundamental Building Block)

This is where the core computation of the transformer happens.

-   **Purpose**: To refine the sequence of patch embeddings by allowing them to exchange information (self-attention) and incorporate guidance from the conditioning latents (cross-attention).
-   **Inputs**:
    -   `hidden_states` (`torch.Tensor`): The sequence of patch embeddings from the previous layer. Shape: `(batch_size, num_patches, hidden_size)`.
    -   `encoder_hidden_states` (`torch.Tensor`): The projected conditioning latent sequence.
    -   `temb` (`torch.Tensor`): The combined time and caption embedding for adaptive normalization.
    -   `image_rotary_emb` (`torch.Tensor`): Rotary position embeddings for the self-attention mechanism.
-   **Internal Modules**:
    -   `norm1`: An adaptive RMS Normalization layer (`LuminaRMSNormZero`) that normalizes the input `hidden_states` using `temb`. It outputs the normalized state and several gating/scaling factors.
    -   `attn1` (**Self-Attention**): Attends over the patch embedding sequence. It uses rotary position embeddings (`image_rotary_emb`) to incorporate spatial information. This allows different patches to understand their relative positions.
    -   `norm1_context`: A standard RMS Normalization layer for the `encoder_hidden_states`.
    -   `attn2` (**Cross-Attention**): The key mechanism for conditioning. The queries are derived from the patch embeddings (`hidden_states`), while the keys and values are derived from the conditioning latents (`encoder_hidden_states`). This allows the model to inject information from the conditioning signal into the generation process.
    -   `gate`: A learnable parameter that scales the output of the cross-attention, allowing the model to dynamically control the strength of the conditioning.
    -   `feed_forward`: A standard feed-forward network (FFN) with a residual connection.
-   **Processing**:
    1.  The input `hidden_states` are normalized using `temb` (adaptive normalization).
    2.  **Self-Attention**: The normalized patch embeddings are passed through `attn1`. Rotary position embeddings are applied to the queries and keys to preserve spatial relationships.
    3.  **Cross-Attention**: The conditioning latents (`encoder_hidden_states`) are normalized. Then, `attn2` computes attention where the queries come from the (normalized) patch embeddings and keys/values come from the (normalized) conditioning latents.
    4.  The outputs of self-attention and the gated cross-attention are summed.
    5.  The first residual connection is applied: the original `hidden_states` are added to the attention output.
    6.  The result is passed through a `feed_forward` network, followed by the second residual connection.
-   **Output**:
    -   `hidden_states` (`torch.Tensor`): The refined sequence of patch embeddings. Shape: `(batch_size, num_patches, hidden_size)`.

---
## 3. Summary of Key Architectural Concepts

-   **Transformer Backbone**: The model uses a deep stack of transformer blocks, which is effective at modeling long-range dependencies in sequence data (in this case, a sequence of image patches).
-   **Adaptive Layer Norm (`adaLN`)**: The `temb` (time and conditioning embedding) is used to dynamically scale and shift the activations within the normalization layers (`LuminaRMSNormZero`, `LuminaLayerNormContinuous`). This is the primary mechanism for feeding the diffusion timestep and conditioning context into the model.
-   **Cross-Attention for Conditioning**: This is the most critical feature for guidance. It allows the model to "look at" the `z_latents` at every block, ensuring the generated output is faithful to the conditioning input.
-   **Rotary Position Embeddings (RoPE)**: Instead of fixed positional embeddings, RoPE is used to encode the relative positions of patches, which is more flexible for varying input sizes and generally performs well.
-   **Patch-Based Processing**: Like Vision Transformers (ViT) and other DiT models, the input latent is treated as a collection of patches, which are processed as a sequence.
