import torch
import torch.nn as nn

class CrossAttentionTransformerLayer(nn.Module):
    """
    Represents a single layer of Transformer-style cross-attention fusion.
    Includes Cross-Attention, Feed-Forward Network, LayerNorms, and Residual Connections.
    """
    def __init__(self, d_query, d_kv, num_heads, dropout_rate=0.1):
        super().__init__()

        # --- Cross Attention Sub-layer ---
        # Pre-Norm for inputs (Query comes from clip, Key/Value from spatial)
        self.query_norm = nn.LayerNorm(d_query)
        self.kv_norm = nn.LayerNorm(d_kv) # Norm for K/V source

        # Projection to attention dimension
        self.query_proj = nn.Linear(d_query, d_query)
        self.key_proj = nn.Linear(d_kv, d_query)
        self.value_proj = nn.Linear(d_kv, d_query)

        # Multihead Cross Attention
        # Note: d_clip must be divisible by num_heads if d_attn == d_clip,
        # but PyTorch's MultiheadAttention uses embed_dim (d_attn here) for its internal projections.
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_query, num_heads=num_heads, dropout=dropout_rate, batch_first=True)

        # Dropout after attention
        self.attn_dropout = nn.Dropout(dropout_rate)

        # Projection after attention (from d_attn back to d_clip)
        # Needed if d_attn is different from d_clip, or just as an output projection.
        # Let's assume FFN input is d_clip, so project back here.
        self.attn_out_proj = nn.Linear(d_query, d_query)

        # --- Feed-Forward Network Sub-layer ---
        # LayerNorm before FFN (applied to output of first residual connection)
        self.ffn_norm = nn.LayerNorm(d_query)

        # Feed-forward network (standard implementation)
        # Input and output dimension is d_clip
        ffn_hidden_dim = 4 * d_query # Common practice
        self.ffn = nn.Sequential(
            nn.Linear(d_query, ffn_hidden_dim),
            # Consider nn.GELU() as used in many modern transformers
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout inside FFN
            nn.Linear(ffn_hidden_dim, d_query)
        )

        # Dropout after FFN
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, query_features, kv_features):
        """
        Args:
            query_features: Features to be updated (e.g., from CLIP) [B, N_q, D_clip]
            kv_features: Features to attend to (e.g., from Spatial Encoder) [B, N_kv, D_spatial_encoder]
                         Note: N_q and N_kv can be different.

        Returns:
            updated_query_features: Processed query features [B, N_q, D_clip]
        """
        # Store original query for residual connection 1
        residual1 = query_features

        # --- Cross Attention Sub-layer ---
        query_norm = self.query_norm(query_features)
        kv_norm = self.kv_norm(kv_features) # Normalize K/V source

        # Project to Q, K, V
        q = self.query_proj(query_norm)     # [B, N_q, D_attn]
        k = self.key_proj(kv_norm)         # [B, N_kv, D_attn]
        v = self.value_proj(kv_norm)       # [B, N_kv, D_attn]

        # Perform cross-attention
        # Output: [B, N_q, D_attn]
        attn_output, _ = self.cross_attention(query=q, key=k, value=v)

        # Project back to d_clip dimension and apply dropout
        attn_output = self.attn_dropout(self.attn_out_proj(attn_output)) # [B, N_q, D_clip]

        # Residual connection 1 (Add & Norm style - common in decoders)
        # Or Pre-Norm style: Norm(X + Dropout(Sublayer(Norm(X)))) -> We are closer to Pre-Norm
        # Let's follow the structure implied by TransformerFusion provided earlier:
        # Attention -> Dropout -> Residual -> Norm -> FFN -> Dropout -> Residual
        query_features = residual1 + attn_output # Add after dropout

        # Store for residual connection 2
        residual2 = query_features

        # --- Feed-Forward Network Sub-layer ---
        # Apply LayerNorm before FFN
        ffn_input = self.ffn_norm(query_features)

        # Apply FFN and dropout
        ffn_output = self.ffn_dropout(self.ffn(ffn_input))

        # Residual connection 2
        updated_query_features = residual2 + ffn_output

        return updated_query_features

class MultiLayerCrossAttentionFusion(nn.Module):
    """
    Stacks multiple CrossAttentionTransformerLayer blocks for deeper fusion.
    """
    def __init__(self, num_layers, d_query, d_kv, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Create a ModuleList to hold all the layers
        self.layers = nn.ModuleList([
            CrossAttentionTransformerLayer(
                d_query=d_query,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        # Optional: Final Layer Normalization after all layers are processed
        # self.final_norm = nn.LayerNorm(d_clip)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: Initial query features [B, N_q, D_clip]
            spatial_encoder_features: Key/Value features (context) [B, N_kv, D_spatial_encoder]
                                     These features remain the same across all layers.
        Returns:
            fused_features: Processed query features after passing through all layers [B, N_q, D_clip]
        """
        # The query features get updated sequentially through the layers
        current_query_features = clip_features

        # Pass the features through each layer
        for layer in self.layers:
            # The K/V features (spatial_encoder_features) act as constant context for each layer
            current_query_features = layer(
                query_features=current_query_features,
                kv_features=spatial_encoder_features # Pass the *original* spatial features to each layer
            )

        # Optional: Apply final normalization
        # fused_features = self.final_norm(current_query_features)
        fused_features = current_query_features # Without final norm

        return fused_features

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration (Example values)
    num_fusion_layers = 4 # Number of cross-attention layers to stack
    clip_dim = 1152       # Dimension of CLIP features (Query)
    spatial_dim = 768     # Dimension of Spatial Encoder features (Key/Value)
    attn_dim = 1152       # Attention dimension (can be same as clip_dim or different)
    num_attn_heads = 18   # Number of attention heads (attn_dim must be divisible by this)
    dropout = 0.1

    # Create the multi-layer fusion module
    multi_layer_fusion = MultiLayerCrossAttentionFusion(
        num_layers=num_fusion_layers,
        d_clip=clip_dim,
        d_spatial_encoder=spatial_dim,
        d_attn=attn_dim,
        num_heads=num_attn_heads,
        dropout_rate=dropout
    )

    # Example Input Tensors
    batch_size = 4
    num_clip_tokens = 100   # Example sequence length for CLIP features
    num_spatial_tokens = 50 # Example sequence length for Spatial features

    clip_input = torch.randn(batch_size, num_clip_tokens, clip_dim)
    spatial_input = torch.randn(batch_size, num_spatial_tokens, spatial_dim)

    # Forward pass
    output_features = multi_layer_fusion(clip_input, spatial_input)

    # Check output shape
    print("Input CLIP shape:", clip_input.shape)
    print("Input Spatial shape:", spatial_input.shape)
    print("Output Fused shape:", output_features.shape)

    # Verify output dimension matches clip_dim
    assert output_features.shape == (batch_size, num_clip_tokens, clip_dim)
    print("Multi-layer cross-attention fusion executed successfully.")
