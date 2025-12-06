import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Simple MLP block: Linear -> Activation -> Dropout -> Linear """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # Dropout is often placed after the first Linear + Activation or after the second Linear
        # Let's place it after the activation here.
        self.dropout1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(drop) # Optional: Add dropout after the second linear layer too

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x) # Apply dropout after second linear
        return x

class CrossAttentionFusionWithMLP(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, mlp_ratio=4.0, proj_drop=0.1):
        """
        Args:
            d_clip: Dimension of clip features.
            d_spatial_encoder: Dimension of spatial encoder features.
            d_attn: Dimension of the attention embedding space (also output of MLPs).
            num_heads: Number of attention heads.
            mlp_ratio: Determines the hidden dimension of the MLPs (hidden_dim = d_attn * mlp_ratio).
            proj_drop: Dropout rate for the projection MLPs and the final output projection.
        """
        super(CrossAttentionFusionWithMLP, self).__init__()

        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)

        # MLP projections
        mlp_hidden_dim = int(d_attn * mlp_ratio)

        self.clip_query_proj = MLP(in_features=d_clip,
                                   hidden_features=mlp_hidden_dim,
                                   out_features=d_attn,
                                   act_layer=nn.GELU,
                                   drop=proj_drop)

        self.spatial_encoder_key_proj = MLP(in_features=d_spatial_encoder,
                                            hidden_features=int(d_attn * mlp_ratio), # Can use a different ratio if desired
                                            out_features=d_attn,
                                            act_layer=nn.GELU,
                                            drop=proj_drop)

        self.spatial_encoder_value_proj = MLP(in_features=d_spatial_encoder,
                                              hidden_features=int(d_attn * mlp_ratio), # Usually same structure as Key MLP
                                              out_features=d_attn,
                                              act_layer=nn.GELU,
                                              drop=proj_drop)

        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        # --- Post-Attention Processing ---
        # Option 1: Keep original post-attention structure (Norm -> OutProj -> Residual -> Dropout)
        # self.out_norm = nn.LayerNorm(d_attn) # Applied to the output of MHA *before* out_proj
        # self.out_proj = nn.Linear(d_attn, d_clip)

        # Option 2: More standard Transformer block structure (MHA -> Dropout -> Residual -> Norm -> FFN(MLP) -> Dropout -> Residual -> Norm)
        # Let's stick closer to the original structure provided for the post-attention part,
        # but note the LayerNorm placement might differ from standard Transformer blocks.

        # Based on the *original* code's post-attention structure:
        self.out_norm = nn.LayerNorm(d_clip) # NOTE: Original code had LayerNorm *after* out_proj, operating on d_clip. Let's correct it based on common practice to apply norm *before* residual or *after* residual+dropout
                                             # Applying LayerNorm on d_attn (MHA output) before out_proj is also common. Let's try that first.
        self.out_norm_attn = nn.LayerNorm(d_attn) # Norm applied to MHA output
        self.out_proj = nn.Linear(d_attn, d_clip)
        self.dropout = nn.Dropout(proj_drop) # Reusing proj_drop, originally 0.1


    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N_spatial, D_spatial_encoder] (Note: sequence length N might differ)
        Returns:
            fused_features: [B, N, D_clip]
        """
        B, N_clip, D_clip = clip_features.shape
        B, N_spatial, D_spatial = spatial_encoder_features.shape

        # Store original clip features for residual connection
        residual = clip_features

        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N_clip, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N_spatial, D_spatial]

        # MLP projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N_clip, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N_spatial, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N_spatial, D_attn]

        # cross attention
        # Query: clip_features projected [B, N_clip, D_attn]
        # Key: spatial_features projected [B, N_spatial, D_attn]
        # Value: spatial_features projected [B, N_spatial, D_attn]
        attn_output, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj,
            average_attn_weights=True # Keep default or set explicitly if needed
        ) # attn_output shape: [B, N_clip, D_attn], attn_weights shape: [B, N_clip, N_spatial] (assuming average_attn_weights=True)

        # --- Post-Attention Processing ---
        # Apply LayerNorm to the attention output (d_attn dimension)
        attn_output_norm = self.out_norm_attn(attn_output)

        # Projection back to D_clip dimension
        fused_features_proj = self.out_proj(attn_output_norm)  # [B, N_clip, D_clip]

        # Apply dropout
        fused_features_dropped = self.dropout(fused_features_proj)

        # Residual connection
        fused_features = residual + fused_features_dropped  # [B, N_clip, D_clip]

        # Final LayerNorm? (Often done *after* residual in Post-LN, or skip if using Pre-LN consistently)
        # The original code had a LayerNorm *here* (self.out_norm(fused_features)). Let's omit for now, assuming Pre-LN style.
        # If you want Post-LN: fused_features = self.out_norm(fused_features) # self.out_norm would need to be LayerNorm(d_clip)

        return fused_features, attn_weights
