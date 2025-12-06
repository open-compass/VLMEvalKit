import torch
import torch.nn as nn
import re
from .cross_attention_transformers import MultiLayerCrossAttentionFusion
from .cross_attention_mlp import CrossAttentionFusionWithMLP
from .video_3d_llm_block import video_3d_llm_fusion_block

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()

        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)

        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)

        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)

        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)

        # dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]

        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]

        # cross attention
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )

        # projection to D_clip dimension
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]

        # residual connection and dropout
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features  # [B, N_clip, D_clip]
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features)

        return fused_features, attn_weights

class TransformerFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads, dropout_rate=0.1):
        super(TransformerFusion, self).__init__()

        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)

        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)

        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)

        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_attn, 4 * d_attn),
            nn.ReLU(),
            nn.Linear(4 * d_attn, d_clip)
        )

        # dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]

        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]

        # cross attention
        attention_output, _ = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )

        # dropout
        attention_output = self.dropout(attention_output)

        # add residual connection
        attention_output_add_residual = attention_output + clip_features

        # pre-norm
        attention_output_add_residual_norm = self.out_norm(attention_output_add_residual)

        # feed-forward network
        feed_forward_output = self.ffn(attention_output_add_residual_norm)   # [B, N_clip, D_clip]

        # dropout
        feed_forward_output = self.dropout(feed_forward_output)

        # add residual connection
        fused_features = feed_forward_output + attention_output_add_residual

        return fused_features

class llava_3d_fusion_block(nn.Module):
    def __init__(self, patch_size=14, latent_dim=1152):
        super(llava_3d_fusion_block, self).__init__()
        self.patch_size = patch_size
        self.points_enc = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
            )

    def forward(self, clip_features, points):
        # points shape: [B, H, W, 3]
        B, H, W, _ = points.shape

        # 1. 获取patch中心点
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # 重塑points为patches
        patches = points.view(B, patch_h, self.patch_size, patch_w, self.patch_size, 3)

        # 计算每个patch的中心点
        patch_center_points = patches.mean(dim=(2, 4))  # [B, patch_h, patch_w, 3]

        # 2. 编码中心点
        encoded_centers = self.points_enc(patch_center_points)  # [B, patch_h, patch_w, latent_dim]

        # 3. 将编码后的特征与clip_features融合
        # 确保维度匹配
        encoded_centers = encoded_centers.view(B, patch_h * patch_w, -1)

        # 直接相加融合
        fused_features = clip_features + encoded_centers

        return fused_features

class MLPFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder, fusion_block_type):
        super(MLPFusion, self).__init__()
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", fusion_block_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))

            def create_mlp():
                modules = [nn.Linear(d_spatial_encoder, d_llm)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(d_llm, d_llm))
                return nn.Sequential(*modules)

            self.spatial_features_mlp = create_mlp()

    def forward(self, clip_features, spatial_features):
        # project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_mlp(spatial_features)

        # add residual connection
        fused_features = clip_features + projected_spatial_features

        return fused_features

class ConcatMLPFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder):
        super(ConcatMLPFusion, self).__init__()

        # MLP to project spatial features (similar to mlp2x_gelu)
        self.spatial_features_proj_mlp = nn.Sequential(
            nn.Linear(d_spatial_encoder, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )

        # MLP for fused features after concatenation
        # Input dim: 2 * d_llm, Output dim: d_llm
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_llm, 4 * d_llm),
            nn.GELU(),
            nn.Linear(4 * d_llm, d_llm)
        )

    def forward(self, clip_features, spatial_features):
        # Project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_proj_mlp(spatial_features) # [B, N, D_llm]

        # Concatenate features
        concatenated_features = torch.cat([clip_features, projected_spatial_features], dim=-1) # [B, N, 2 * D_llm]

        # Pass through fusion MLP
        fused_features = self.fusion_mlp(concatenated_features) # [B, N, D_llm]

        return fused_features

class ConcatSelfAttentionFusion(nn.Module):
    def __init__(self, d_llm, d_spatial_encoder, num_heads, dropout_rate=0.1):
        super(ConcatSelfAttentionFusion, self).__init__()

        # MLP to project spatial features
        self.spatial_features_proj_mlp = nn.Sequential(
            nn.Linear(d_spatial_encoder, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )

        # Pre-attention normalization
        self.pre_attn_norm = nn.LayerNorm(2 * d_llm)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=2 * d_llm,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Post-attention normalization
        self.post_attn_norm = nn.LayerNorm(2 * d_llm)

        # Final linear projection
        self.output_proj = nn.Linear(2 * d_llm, d_llm)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, clip_features, spatial_features):
        # Project spatial features to llm dimension
        projected_spatial_features = self.spatial_features_proj_mlp(spatial_features) # [B, N, D_llm]

        # Concatenate features
        concatenated_features = torch.cat([clip_features, projected_spatial_features], dim=-1) # [B, N, 2 * D_llm]

        # Pre-attention normalization
        normed_features = self.pre_attn_norm(concatenated_features)

        # Self-attention
        attn_output, _ = self.self_attention(
            query=normed_features,
            key=normed_features,
            value=normed_features
        ) # [B, N, 2 * D_llm]

        # Dropout and residual connection 1
        attn_output = self.dropout(attn_output)
        attn_output_residual = attn_output + concatenated_features # Residual connection before final projection

        # Post-attention normalization
        normed_attn_output = self.post_attn_norm(attn_output_residual)

        # Final projection
        fused_features = self.output_proj(normed_attn_output) # [B, N, D_llm]

        return fused_features

def build_multimodal_fusion_block(config, delay_load=False, **kwargs):
    fusion_block_type = getattr(config, "fusion_block", "cross_attention")
    d_clip = config.mm_hidden_size
    d_llm = config.hidden_size
    d_attn = d_clip
    d_spatial_encoder = getattr(config, "spatial_feature_dim", 768)
    if fusion_block_type == "cross_attention_with_mlp":
        return CrossAttentionFusionWithMLP(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            mlp_ratio=4.0,
            proj_drop=0.1
        )
    elif fusion_block_type == "cross_attention":
        return CrossAttentionFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18
        )
    elif fusion_block_type == "mlp_after_clip_proj":
        return MLPFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder,
            fusion_block_type="mlp2x_gelu"
        )
    elif fusion_block_type == "transformer":
        return TransformerFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18,
            dropout_rate=0.1
        )
    elif fusion_block_type == "concat_mlp":
        return ConcatMLPFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder
        )
    elif fusion_block_type == "concat_self_attention":
        return ConcatSelfAttentionFusion(
            d_llm=d_llm,
            d_spatial_encoder=d_spatial_encoder,
            num_heads=36,
            dropout_rate=0.1
        )
    elif fusion_block_type == "llava_3d_fusion_block":
        return llava_3d_fusion_block(
            patch_size=16,
            latent_dim=1152
        )
    elif fusion_block_type == "video_3d_llm_fusion_block":
        return video_3d_llm_fusion_block(
            patch_size=14,
            latent_dim=d_llm
        )
    elif fusion_block_type.endswith("_layer_cross_attention"):
        num_layers = int(fusion_block_type.split("_")[0])
        return MultiLayerCrossAttentionFusion(
            num_layers=num_layers,
            d_query=d_llm,
            d_kv=d_spatial_encoder,
            num_heads=64
        )
    raise ValueError(f"Unknown fusion block type: {fusion_block_type}")
