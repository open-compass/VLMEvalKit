import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 将类名修改为 video_3d_llm_fusion_block
class video_3d_llm_fusion_block(nn.Module):
    def __init__(self, patch_size=14, latent_dim=3584, temperature=10000):
        """
        初始化融合块。

        Args:
            patch_size (int): 图像块的大小 (假设 H 和 W 都能被 patch_size 整除)。
            latent_dim (int): 视觉特征和位置编码的目标维度。
            temperature (int): 正弦编码中的温度参数。
        """
        # 同时修改 super() 中的类名
        super(video_3d_llm_fusion_block, self).__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.temperature = temperature
        # 不再需要 MLP 编码器

    def sinusoidal_3d_encoding(self, coords, embedding_dim):
        """
        计算 3D 坐标的正弦位置编码，处理 embedding_dim 不能被 3 整除的情况。

        Args:
            coords (torch.Tensor): 输入坐标，形状 [B, N, 3]。
            embedding_dim (int): 输出编码的目标维度 (例如 3584)。

        Returns:
            torch.Tensor: 位置编码，形状 [B, N, embedding_dim]。
        """
        B, N, _ = coords.shape
        # 不再需要 assert embedding_dim % 6 == 0

        # 计算每个坐标维度分配的基础特征数 (整数除法)
        num_feats = embedding_dim // 3
        if num_feats == 0: # 如果维度太小，无法分配
            # 返回零向量或者抛出错误，这里选择返回零
            print("Warning: embedding_dim is too small for sinusoidal 3D encoding.")
            return torch.zeros((B, N, embedding_dim), dtype=coords.dtype, device=coords.device)

        # 确保用于 sin/cos 的维度数 >= 2
        num_sin_cos_pairs = num_feats // 2
        if num_sin_cos_pairs == 0:
            print("Warning: num_feats per dimension is less than 2, cannot perform sin/cos encoding properly.")
            # 可以选择仅使用一个频率或返回零
            # 为了简单，这里也返回零，但在实际应用中可能需要更复杂的处理
            return torch.zeros((B, N, embedding_dim), dtype=coords.dtype, device=coords.device)


        # 创建用于计算频率的维度张量
        dim_t = torch.arange(num_sin_cos_pairs, dtype=torch.float32, device=coords.device)
        # 注意这里除数是 num_sin_cos_pairs，对应原始代码中的 num_feats // 2
        dim_t = self.temperature ** (2 * dim_t / num_sin_cos_pairs) # [num_feats/2]

        # 准备坐标 [B, N, 3, 1]
        pos_input = coords.unsqueeze(-1) / dim_t # [B, N, 3, num_feats/2]

        # 计算 sin 和 cos 编码 [B, N, 3, num_feats/2, 2] -> [B, N, 3, num_feats]
        # 使用 flatten(3) 来交错 sin 和 cos
        pos_emb_pairs = torch.stack((pos_input.sin(), pos_input.cos()), dim=4).flatten(3) # Shape: [B, N, 3, num_feats]

        # 如果 num_feats 是奇数，原始代码会特殊处理，这里我们简化一下
        # 我们的 pos_emb_pairs 维度是 num_feats (偶数)

        # 拼接 x, y, z 维度的编码
        # calculated_dim = 3 * num_feats # 实际计算出的维度
        pos = pos_emb_pairs.flatten(2) # [B, N, 3 * num_feats]
        calculated_dim = pos.shape[-1] # 获取实际计算出的维度

        # 创建目标维度的零张量
        final_emb = torch.zeros((B, N, embedding_dim), dtype=coords.dtype, device=coords.device)

        # 将计算出的 pos 填充到前面
        final_emb[:, :, :calculated_dim] = pos

        return final_emb

    def forward(self, clip_features, points):
        """
        前向传播。

        Args:
            clip_features (torch.Tensor): 视觉特征，形状 [B, num_patches, latent_dim]。
            points (torch.Tensor): 密集 3D 坐标，形状 [B, 3, H, W]。

        Returns:
            torch.Tensor: 融合后的特征，形状 [B, num_patches, latent_dim]。
        """
        # points shape: [B, 3, H, W]
        B, _, H, W = points.shape
        num_patches_in = clip_features.shape[1]

        # 1. 获取patch中心点（或平均点）
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        num_patches = patch_h * patch_w
        resized_height = patch_h * self.patch_size
        resized_width = patch_w * self.patch_size
        # resize points to [B, 3, resized_height, resized_width]
        points = F.interpolate(points, size=(resized_height, resized_width), mode='nearest')

        # 确保计算出的块数量与输入的 clip_features 匹配
        assert num_patches == num_patches_in, \
               f"Number of patches mismatch: calculated {num_patches} vs input {num_patches_in}"

        # 重塑points为patches
        # [B, 3, patch_h, patch_size, patch_w, patch_size]
        patches = points.view(B, 3, patch_h, self.patch_size, patch_w, self.patch_size)

        # 计算每个patch的中心点（平均坐标）
        # [B, 3, patch_h, patch_w]
        patch_center_points = patches.mean(dim=(3, 5))

        # 重塑为 [B, num_patches, 3] 以便编码
        patch_center_points_flat = patch_center_points.view(B, 3, num_patches).permute(0, 2, 1)

        # 2. 使用正弦函数编码中心点
        # encoded_centers: [B, num_patches, latent_dim]
        encoded_centers = self.sinusoidal_3d_encoding(patch_center_points_flat, self.latent_dim)

        # 3. 将编码后的特征与clip_features融合（直接相加）
        fused_features = clip_features + encoded_centers

        return fused_features

# --- 示例用法 (也更新了类名) ---
if __name__ == '__main__':
    batch_size = 2
    height, width = 224, 224 # 示例图像尺寸
    patch_size = 14
    latent_dim = 768 # 示例维度 (需要能被6整除)

    # 模拟输入
    num_patches = (height // patch_size) * (width // patch_size)
    mock_clip_features = torch.randn(batch_size, num_patches, latent_dim)
    mock_points = torch.randn(batch_size, height, width, 3) * 10 # 模拟世界坐标

    # 创建模块实例 (使用新名称)
    fusion_block = video_3d_llm_fusion_block(patch_size=patch_size, latent_dim=latent_dim)

    # 执行前向传播
    fused_output = fusion_block(mock_clip_features, mock_points)

    print("输入 clip_features 形状:", mock_clip_features.shape)
    print("输入 points 形状:", mock_points.shape)
    print("输出 fused_features 形状:", fused_output.shape)
    # 预期输出形状: [2, 256, 768] (对于 224x224 输入和 14x14 patch)
