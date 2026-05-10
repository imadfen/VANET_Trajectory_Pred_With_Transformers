# Instruction Prompt for Implementing the Dual-Loop VANET Framework

*The following instructions were used to implement the Dual-Loop and Intent architecture straight into the Transformer codebase.*

## 1. CLI Attributes (`src/options.py`)
Add the following configurable variables to the command line setup so they can be tuned later:
- `num_intents` (default: 4)
- `intent_weight` (default: 0.0)
- `entropy_threshold` (default: 0.5)
- `beacon_hz_low` (default: 2.0)
- `beacon_hz_high` (default: 10.0)
- `relay_constant` (default: 0.1)
- `relay_min_wait_ms` (default: 1.0)
- `relay_max_wait_ms` (default: 100.0)

## 2. Loop A: Adaptive Awareness (`src/loops/loop_a.py`)
Create a new python module that calculates discrepancy.
* **Component 1: DiscrepancyMonitor**
  - Implement a `check()` function taking two sequences (Transformer's `predicted_window` and actual sensor `actual_window`, both `(seq_len, feat_dim)`).
  - Calculate `residual` as the mean $L_2$ norm representing the discrepancy between both arrays.
  - Implement logic comparing `residual` to `epsilon_threshold`:
    - **Low Entropy Mode** (`residual < epsilon`): Return `Flag=0`, `residual`, and `beacon_hz = beacon_hz_low` (2Hz).
    - **High Entropy Mode**: Return `Flag=1`, `residual`, and `beacon_hz = beacon_hz_high` (10Hz).
* **Component 2: HallucinationEngine**
  - Create a stub wrapper that accepts a frozen Transformer encoder.
  - Implement an `hallucinate()` function which takes a context window array and autoregressively generates the next $N$ future steps for neighbors to use when the ego vehicle falls silent to 2Hz mode.

## 3. Loop B: Selective Forwarding (`src/loops/loop_b.py`)
Create a new module handling MAC backoff algorithms based on Transformer intent.
* **Component 1: StabilityScorer**
  - Take the raw `intent_logits` (1D array of 4 probabilities).
  - Convert logits via Softmax to obtain $P_{stable}$ (the probability for the 'Maintain Lane' index 0).
* **Component 2: MACBiasMapper**
  - Implement a mapping formula mapping $P_{stable}$ into a network `T_wait_ms`.
  - Use logarithmic mapping bounded by `min_wait_ms` and `max_wait_ms` OR raw $T_{wait} = (\text{constant} / P_{stable}) \times 1000$.
  - Return the final clipped millisecond delay ensuring unstable cars wait ~100ms and stable cars wait ~1ms.

## 4. Structural Model Modifications (`src/transformer_model/encoder.py`)
Modify the `TSTransformerEncoder` and its internal layers to support the "Dual-Head" and Attention exports.
* **Intent Classification Head**: 
  - Add `self.intent_head = nn.Linear(embedding_dim, num_intents)` alongside the typical output regressor head. 
  - In `forward()`, mean-pool the sequence output (excluding padded tokens) and pipe it through the new `intent_head` to output a `(B, num_intents)` tensor.
* **Attention Map Extraction**:
  - Update `TransformerBatchNormEncoderLayer` to pass a custom `return_attn=True` down into the underlying multi-head attention module.
  - In `TSTransformerEncoder.forward()`, add an empty `attn_maps` list. If `return_attn` is True, manually step through the encoder layers individually and capture each layer's `(B, T, T)` attention output tensor matrix. 
  - Change the global `forward()` signature to return 4 objects: `(recon_output, intent_logits, attn_maps, embeddings)`.

## 5. Main Model Integration (`src/transformer_model/model.py`)
Update the primary training and inference wrappers to utilize the new structural changes.
* **Dual-Loss Function**: 
  - Unpack the new 4 outputs from the `encoder()`.
  - Check if `intent_weight > 0` and if labels are attached. If so, apply `torch.nn.functional.cross_entropy` classification loss and add it to the original MSE imputation loss. 
* **Variable Passthrough**: 
  - Update `create_model()` to pipe the `intent_weight` down into `UnsupervisedAttentionModel`.
  - Update `val_evaluator` to save `intent_logits` and `attn_maps` inside `per_batch` so offline metrics can extract them for simulating Loop A / Loop B in Jupyter notebooks later.
