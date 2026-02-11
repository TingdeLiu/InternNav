import math

import torch
import torch.nn as nn

from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


"""
该模块实现了用于导航任务的多种视觉编码器主干网络（Backbones）。
主要功能包括：
1. 位置编码：提供了正弦余弦编码 (SinusoidalPosEmb)、固定位置编码 (PositionalEncoding) 和可学习位置编码 (LearnablePositionalEncoding)。
2. Token 压缩：TokenCompressor 使用交叉注意力机制将变长 Token 序列压缩为固定长度。
3. 视觉主干网络：
   - DAT_RGBD_Patch_Backbone & RGBDBackbone: 基于 DepthAnythingV2 的 RGBD 融合编码器，用于提取环境的时空特征。
   - ImageGoalBackbone: 用于处理图像目标的编码器，支持多通道输入融合。
   - PixelGoalBackbone: 像素级目标编码器，支持自定义通道数的特征提取。
这些模块共同构成了导航模型中处理视觉感知、深度感知和目标引导的核心组件。
"""


class SinusoidalPosEmb(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PositionalEncoding(nn.Module):
    """固定位置编码模块"""

    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[: x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码，使用 nn.Embedding 实现"""

    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """
        前向传播
        x: 输入的 token 向量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_encoding = self.position_embedding(position_ids)
        return position_encoding


class TokenCompressor(nn.Module):
    """Token 压缩器，使用交叉注意力将变长序列压缩为固定长度的特征"""
    def __init__(self, embed_dim, num_heads, target_length):
        super(TokenCompressor, self).__init__()
        self.target_length = target_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 可学习的目标序列，作为 Query
        self.target_embedding = nn.Embedding(target_length, embed_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)

        # 多头注意力层，用于跨注意力操作
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, padding_mask=None):
        """
        前向传播
        x: (bs, N, 384) - 输入序列（长度可变）
        padding_mask: (bs, N) - 输入序列的掩码（True 表示填充位置）
        """
        bs, token_len, _ = x.shape

        # 为输入 token 添加位置编码
        token_pe = self.token_positional_encoding(x)
        x = x + token_pe

        # 扩展目标 Embedding 作为 Query
        query = self.target_embedding.weight.unsqueeze(0).expand(bs, -1, -1)

        # 为 Query 添加位置编码
        query_pe = self.query_positional_encoding(query)

        query = query + query_pe

        # 跨注意力机制：目标 Embedding 是 Query，输入 x 是 Key 和 Value
        out, _ = self.cross_attention(query=query, key=x, value=x, key_padding_mask=padding_mask)
        return out


class DAT_RGBD_Patch_Backbone(nn.Module):
    """基于 DepthAnythingV2 的 RGBD Patch 主干网络"""
    def __init__(
        self,
        image_size=224,
        embed_size=512,
        finetune=True,
        memory_size=8,
        checkpoint="checkpoints/depth_anything_v2_vits.pth",
        input_dtype="bf16",
        version=0.0,
        device='cuda:0',
    ):
        super().__init__()
        self.finetune = finetune
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.input_dtype = torch.bfloat16 if input_dtype == "bf16" else torch.float32
        self.version = version

        # 模型配置：使用 vits 版本的 DepthAnythingV2
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.rgb_model = self.rgb_model.pretrained

        # 图像预处理参数 (ImageNet 标准)
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.input_dtype)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=self.input_dtype)

        if finetune:
            self.rgb_model.train()
        else:
            self.rgb_model.eval()

        # 深度图模型也使用 DepthAnythingV2 的编码器
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained
        self.depth_model.train()

        # Transformer 解码器的 Query 和位置编码
        self.former_query = nn.Embedding(self.memory_size * 16, 384)
        nn.init.constant_(self.former_query.weight, val=0)

        if self.version > 0.0:
            self.former_pe = nn.Embedding((self.memory_size * 2) * 256, 384)
        else:
            self.former_pe = nn.Embedding((self.memory_size + 1) * 256, 384)

        nn.init.constant_(self.former_pe.weight, val=0)
        
        # 使用 TransformerDecoder 进行时空特征融合
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)

    def forward(self, images, depths):
        """
        前向传播
        images: RGB 图像张量
        depths: 深度图张量
        """
        # 处理 RGB 图像
        if len(images.shape) == 4:
            tensor_images = images.to(dtype=self.input_dtype).permute(0, 3, 1, 2)
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (
                tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(images.device)
            ) / self.preprocess_std.to(images.device).reshape(1, 3, 1, 1)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
        elif len(images.shape) == 5:
            B, T, H, W, C = images.shape
            tensor_images = images.to(dtype=self.input_dtype).permute(0, 1, 4, 2, 3)
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (
                tensor_images - self.preprocess_mean.to(images.device).reshape(1, 3, 1, 1)
            ) / self.preprocess_std.to(images.device).reshape(1, 3, 1, 1)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B, T * 256, -1)

        if not self.finetune:
            image_token = image_token.detach()

        # 处理深度图
        if len(depths.shape) == 4:
            tensor_depths = depths.to(dtype=self.input_dtype).permute(0, 3, 1, 2)
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            # 深度图复制为 3 通道以适配编码器
            tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(depths.shape) == 5:
            B, T, H, W, C = depths.shape
            tensor_depths = depths.to(dtype=self.input_dtype).permute(0, 1, 4, 2, 3)
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            tensor_depths = torch.cat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B, T * 256, -1)

        # 生成位置编码索引
        if self.version > 0.0:
            former_pe_indice = torch.arange((self.memory_size * 2) * 256, device=images.device).expand(
                image_token.shape[0], (self.memory_size * 2) * 256
            )
        else:
            former_pe_indice = torch.arange((self.memory_size + 1) * 256, device=images.device).expand(
                image_token.shape[0], (self.memory_size + 1) * 256
            )

        # 连接 RGB 和深度特征并添加位置编码
        former_pe = self.former_pe(former_pe_indice)
        former_token = torch.cat((image_token, depth_token), dim=1) + former_pe

        # 准备解码器 Query
        former_query_indice = torch.arange(self.memory_size * 16, device=images.device).expand(
            image_token.shape[0], self.memory_size * 16
        )
        former_query = self.former_query(former_query_indice)

        # 通过 TransformerDecoder 提取最终特征
        memory_token = self.former_net(former_query, former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token


class RGBDBackbone(nn.Module):
    """通用的 RGBD 主干网络模块"""
    def __init__(
        self,
        image_size=224,
        embed_size=512,
        finetune=True,
        memory_size=8,
        checkpoint="checkpoints/depth_anything_v2_vits.pth",
        device='cuda:0',
    ):
        super().__init__()
        # 确保设备设置有效
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.finetune = finetune
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        
        # 编码器配置
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        
        # TODO: 适配特定的 checkpoint 加载逻辑
        self.rgb_model.load_state_dict(torch.load(checkpoint), strict=False)
        self.rgb_model = self.rgb_model.pretrained.float()
        
        # 预处理参数
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        
        if finetune:
            self.rgb_model.train()
        else:
            self.rgb_model.eval()
            
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        
        # 使用可学习的位置编码
        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)
        self.to(device)

    def forward(self, images, depths):
        """
        前向传播
        images: RGB 图像 (B, H, W, C) 或 (B, T, H, W, C)
        depths: 深度图 (B, H, W, C) 或 (B, T, H, W, C)
        """
        device = self._get_device()
        images = images.to(device)
        depths = depths.to(device)
        
        # 处理 RGB 图像特征提取
        if len(images.shape) == 4:
            tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (
                tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(device)
            ) / self.preprocess_std.reshape(1, 3, 1, 1).to(device)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
        elif len(images.shape) == 5:
            tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
            B, T, C, H, W = tensor_images.shape
            tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
            tensor_norm_images = (
                tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(device)
            ) / self.preprocess_std.reshape(1, 3, 1, 1).to(device)
            image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B, T * 256, -1)
            
        if not self.finetune:
            image_token = image_token.detach()
            
        # 处理深度图特征提取
        if len(depths.shape) == 4:
            tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
        elif len(depths.shape) == 5:
            tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
            B, T, C, H, W = tensor_depths.shape
            tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
            tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
            depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0].reshape(B, T * 256, -1)
            
        # 特征拼接并添加位置编码
        former_token = torch.concat((image_token, depth_token), dim=1) + self.former_pe(
            torch.concat((image_token, depth_token), dim=1)
        )
        
        # Transformer 解码器 Query 生成
        former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=device))
        
        # 特景压缩与融合
        memory_token = self.former_net(former_query, former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token

    def _get_device(self):
        """安全获取模型所在设备"""
        # 尝试通过模型参数获取
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        # 尝试通过缓冲区获取
        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        # 尝试通过子模块获取
        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        # 最后回退到默认设备
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageGoalBackbone(nn.Module):
    """图像目标编码器，用于处理视觉目标"""
    def __init__(self, image_size=224, embed_size=512, device='cuda:0'):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        
        # 修改第一层卷积以适配 6 通道输入（例如：当前图像 + 目标图像）
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=6,
            out_channels=self.imagegoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.imagegoal_encoder.patch_embed.proj.kernel_size,
            stride=self.imagegoal_encoder.patch_embed.proj.stride,
            padding=self.imagegoal_encoder.patch_embed.proj.padding,
        )
        self.imagegoal_encoder.train()
        self.project_layer = nn.Linear(384, embed_size)
        self.to(device)

    def forward(self, images):
        """
        前向传播
        images: 输入张量，形状通常为 (B, C, H, W)，其中 C 可能为 6 (叠加图像)
        """
        assert len(images.shape) == 4  # B,C,H,W
        device = self._get_device()
        images = images.to(device)
        tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        
        # 提取中间层特征并取空间维度平均
        image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token

    def _get_device(self):
        """安全获取模型所在设备"""
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelGoalBackbone(nn.Module):
    """像素级目标编码器，支持自定义通道数"""
    def __init__(self, image_size=224, embed_size=512, pixel_channel=7, device='cuda:0'):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        
        # 修改第一层卷积以适配特定通道数 (pixel_channel)
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=pixel_channel,
            out_channels=self.pixelgoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.pixelgoal_encoder.patch_embed.proj.kernel_size,
            stride=self.pixelgoal_encoder.patch_embed.proj.stride,
            padding=self.pixelgoal_encoder.patch_embed.proj.padding,
        )
        self.pixelgoal_encoder.train()
        self.project_layer = nn.Linear(384, embed_size)
        self.to(device)

    def forward(self, images):
        """
        前向传播
        images: 输入张量，形状通常为 (B, C, H, W)，其中 C 为 pixel_channel
        """
        assert len(images.shape) == 4  # B,C,H,W
        device = self._get_device()
        images = images.to(device)
        tensor_images = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        
        # 提取中间层特征并取空间维度平均
        image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
        image_token = self.project_layer(image_token)
        return image_token

    def _get_device(self):
        """安全获取模型所在设备"""
        try:
            for param in self.parameters():
                return param.device
        except StopIteration:
            pass

        try:
            for buffer in self.buffers():
                return buffer.device
        except StopIteration:
            pass

        for module in self.children():
            try:
                for param in module.parameters():
                    return param.device
            except StopIteration:
                continue

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
