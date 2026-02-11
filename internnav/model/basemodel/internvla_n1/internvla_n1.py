from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

# 轨迹查询 token 在词表中的占位 id，用于填充可训练的轨迹 latent
TRAJ_TOKEN_INDEX = 151667
# 图像占位 token id，用于将视觉特征写回文本序列
IMAGE_TOKEN_INDEX = 151655
# 与 ResNet 预训练保持一致的归一化均值/方差
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    model_type = "internvla_n1"

    def __init__(self, **kwargs):  #通过**kwargs可以在不修改构造函数签名的情况下，随意往里面塞入新的配置信息。
        super().__init__(**kwargs) #继承父类属性
        # 透传来自外部的额外模型配置（包含系统1类型、噪声调度等）
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel): #多重继承，负责“提取特征和理解”
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM): #多重继承，负责“生成文本和轨迹”
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        # 主干多模态模型，整合文本、图像/视频与轨迹查询 token
        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 初始化权重并应用最终的处理步骤
        self.post_init()

        # 注册视觉归一化用的均值/方差，作为 buffer 以便推理和训练共享
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,  # 轨迹起始位置。
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None, #真实的 3D 轨迹数据
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 若未显式传入输入嵌入，则根据 input_ids 构造，并在其中插入视觉/轨迹特征
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids) #文字（token id）通过 embed_tokens 变成向量
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) #图像通过 visual 编码成向量
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item() #图像占位 token 数量
                n_image_features = image_embeds.shape[0] #图像特征数量
                 # 检查图像 token 与视觉特征数量是否匹配
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                # 将图像占位 token 替换成对应视觉编码特征
                mask = input_ids == self.config.image_token_id #图像占位 token 的掩码
                mask_unsqueezed = mask.unsqueeze(-1) #扩展最后一维
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds) #扩展到与输入嵌入相同形状
                image_mask = mask_expanded.to(inputs_embeds.device) #图像掩码
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype) #图像特征
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds) #替换图像占位 token

            # 视频同理，写回视频占位 token
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # 轨迹查询 token 替换为可训练的 latent queries
            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item() #轨迹 token 数量
            traj_idx = input_ids == TRAJ_TOKEN_INDEX #轨迹 token 掩码
            #
            """
            # =====================latent_queries===================
            # 维度为(1, n_query, hidden_size-4096)
            #1.静态阶段 （训练时）
            #在模型训练过程中，latent_queries 确实在不断进化。它通过成千上万次的“挨打”（Loss 反馈），学习到了一些通用的空间逻辑。

            对图像的经验：它学习到“通常图像的下方是地面，上方是天空”，“物体如果变大，通常意味着距离变近”。

            对文本的经验：它学习到当文本出现“左转”或“避障”时，应该关注图像中的哪些特征分布。

            #2.动态阶段 （推理时）
            你把 latent_queries 插入到 inputs_embeds 并送入 Transformer 之后，真正的“学习（提取）”发生了：

            交叉注意力 (Cross-Attention)： 进入 Transformer 层后，latent_queries 会作为 Query，去和图像、文本的 Key/Value 进行计算。

            内容注入：

            它会从图像 Token 中吸取：“当前路面有个大坑，坐标在 (x, y)”。

            它会从文本 Token 中吸取：“用户现在的指令是预测未来 3 秒的轨迹”。

            结论： 训练结束后，latent_queries 变成了一个**“极其专业的面试官”**。它本身不记得具体的简历（图像/文本），但它知道问什么问题
            （Attention）能从简历中瞬间抓取到 3D 预测所需的关键信息。
            """
            latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
            H = latent_queries.shape[-1] #隐藏维度
            latent_queries = latent_queries.contiguous().view(-1, H) #展平为二维
             # 写回轨迹查询 token 位置
            if n_traj_tokens != 0:
                inputs_embeds[traj_idx] = latent_queries

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 如果 attention_mask 为 4D，就无法继续计算 RoPE 偏移。TODO @raushan 需要修复

        # RoPE 偏移 指的是：将视觉 token 的二维位置信息融入到位置编码中
        # position_ids 计算 RoPE 偏移量，仅在预填充阶段计算一次，后续生成阶段复用
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # 仅在预填充阶段为每次生成计算一次 RoPE 索引
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # 否则复用已有 rope_deltas，叠加 cache_position 得到正确的 position_ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # 否则 deltas 只是整数 0
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # 语言模型输出隐藏状态，它是每一个 Token（包括文字、图片块、轨迹查询）在经过全场注意力计算后的“浓缩向量”。
        hidden_states = outputs[0] #最后一层隐藏状态 (batch_size, seq_length, hidden_size)
        logits = self.lm_head(hidden_states) #将隐藏状态翻译成文字概率的过程，线性层得到词表大小的 logits，通过Softmax决定下一个词

        loss = None
        if labels is not None:
            # 根据 t_s_pos 定位轨迹查询段，拼接时间维后与轨迹长度对齐
            traj_hidden_states = []
            for b in range(hidden_states.shape[0]): #batch size
                #这一步把那些已经“看懂了图”的 N 个探测向量从整个句子中抽离了出来。
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            # 根据视频真实帧数构造掩码，忽略 padding 帧的损失
            loss_mask = torch.arange(traj_images.size(1), device=self.device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.unsqueeze(1)

            # Motion Planning
            if 'nextdit' in self.get_system1_type(): #处理rgb图像
                # System1 扩散式轨迹预测（nextdit）
                if 'async' in self.get_system1_type():
                    # 异步：目标帧与当前帧成对送入视觉编码，获得记忆 token 与轨迹条件拼接
                    cur_images = traj_images.flatten(0, 1) #展开为 (bs*select_size, 3, 224, 224)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1) #目标帧复制
                     # 构造图像对 (目标帧 + 当前帧)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3) # 将“目标”与“当前”叠在一起
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std # 归一化
                    
                    # 提取图像对特征
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    #Memory Encoder：它不仅看现在的特征，还把“目标与当前的差异”编码成一种记忆特征。
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    # 将轨迹条件映射到条件空间，与记忆 token 级联作为条件
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = torch.cat([memory_tokens, traj_hidden_states], dim=1) 
                else:
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = traj_hidden_states

                # 对真实轨迹加噪，使用 DiT 预测噪声（flow matching 风格）
                relative_poses = traj_poses.flatten(0, 1) 
                bsz = relative_poses.shape[0] #batch size
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype) #标准正态噪声
                 # 随机采样时间步，获取对应噪声等级
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                #构建带噪轨迹，relative_poses是真实的 3D 轨迹
                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                #预测速度向量，target是模型需要学习的方向
                target = noise - relative_poses
                # 带掩码的 MSE 损失，忽略未用帧
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                masked_loss = loss * mask
                loss = masked_loss.sum() / mask.sum() / (loss.shape[1] * loss.shape[2])

            elif 'navdp' in self.get_system1_type(): #处理RGB-D图
                # System1 导航轨迹预测（navdp）
                if 'async' in self.get_system1_type():
                    # 异步：目标图 + 当前图/深度成对，调用 navdp VLM 分支
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    # 调用 navdp 异步前向接口
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    loss = masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])

            else:
                raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    #推理阶段（Inference） 是如何生成导航条件
    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        # 生成轨迹条件 latent：
        # 1) input_ids 是带有 IMAGE 占位 token 的文本序列，用于指示图像特征写入位置
        # 2) 对 input_ids 做词嵌入（文本部分）
        # 3) 视觉编码写回 IMAGE 占位 token，对齐文本与图像模态
        # 4) 追加 n_query 个轨迹查询占位 token（仅占位，不用其词向量）
        # 5) 拼接可训练的轨迹 latent queries，送入模型前向
        # 6) 截取末尾 n_query 的隐藏状态，作为轨迹条件向量
        input_ids.to(self.get_model().device)
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)

        # 轨迹查询可训练向量，按 batch 复制备用
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()

        # 在文本序列末尾追加 n_query 个轨迹占位 token（只占位，不实际用词向量）
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        # 视觉编码：将图像特征写入 IMAGE 占位 token 位置，实现跨模态对齐
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)
        # 融合层：跨模态对齐
        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        # 将轨迹查询 latent 拼到序列尾部，供后续模型读取
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        # 计算 RoPE 位置编码（含视觉 token 的二维位置信息），再前向推理
        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)

        #思考层：Transformer 的全场调度，尾的那几个 latent_queries 会观察前面的图片特征（看路况）和文字特征（看指令）
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                # attention_mask=attention_mask,  # 如需遮挡可在上游传入
                output_hidden_states=True,
                return_dict=True,
            )
        # 仅保留末尾 n_query 的隐藏状态，作为轨迹条件 latent
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]

        return hidden_states

    #推理阶段（Inference） 是如何生成轨迹
    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32, #每个条件采样多少条轨迹
    ):
        if 'nextdit' in self.get_system1_type():
            # 扩散式采样生成轨迹，支持 classifier-free guidance
            scheduler = FlowMatchEulerDiscreteScheduler()
            device = traj_latents.device
            dtype = traj_latents.dtype

            traj_latents = self.get_model().cond_projector(traj_latents)
            if 'async' in self.get_system1_type():
                # 异步：图像对编码为记忆 token，与轨迹条件拼接
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
                hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
            else:
                hidden_states = traj_latents
            # 构造无条件分支的隐藏状态（全 0 向量）
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
            batch_size = traj_latents.shape[0]
            latent_size = predict_step_nums
            latent_channels = 3

            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

            for t in scheduler.timesteps:
                # 对当前轨迹 latent 编码并加上位置编码
                latent_features = self.get_model().action_encoder(latents)
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
                latent_model_input = latent_features.repeat(2, 1, 1)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # 预测噪声
                noise_pred = self.get_model().traj_dit(
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)

                # classifier-free guidance：条件与无条件分支插值，不看图片和指令（uncond），一个是看着图片和指令（cond）。
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # 执行一步采样更新 x_t -> x_{t-1}
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        elif 'navdp' in self.get_system1_type():
            # 导航预测分支，异步/同步分别调用 navdp 的不同接口
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
