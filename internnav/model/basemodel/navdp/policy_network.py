"""
NavDP 推理用策略网络（从 NavDP 项目迁入）

NavDP_Policy 支持 5 种导航任务模式（pointgoal / nogoal / imagegoal / pixelgoal / mixgoal），
融合 RGBD 场景记忆与多种目标表示后，通过扩散采样生成候选轨迹并用 Critic 评分。
"""

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from internnav.model.basemodel.navdp.policy_backbone import (
    LearnablePositionalEncoding,
    NavDP_ImageGoal_Backbone,
    NavDP_PixelGoal_Backbone,
    NavDP_RGBD_Backbone,
    SinusoidalPosEmb,
)


class NavDP_Policy(nn.Module):
    """NavDP 主策略网络，融合 RGBD 与多种目标表示后用扩散生成轨迹并评分。"""

    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim

        # 输入编码：场景记忆（RGBD）、三维点目标、像素目标、图像目标
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size,token_dim,memory_size=memory_size,device=device)
        self.point_encoder = nn.Linear(3,self.token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size,token_dim,device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size,token_dim,device=device)

        # 融合层：Transformer 解码器，将动作序列作为 tgt，条件特征作为 memory
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)

        self.input_embed = nn.Linear(3,token_dim) # 动作编码，供去噪与 critic 共享
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4) # 条件位置编码：时间+三类目标+记忆
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        # 短步数扩散调度器，用于快速采样轨迹
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                           beta_schedule='squaredcos_cap_v2',
                           clip_sample=True,
                           prediction_type='epsilon')

        # 自回归掩码：强制动作预测只依赖当前及过往动作，防泄漏未来信息
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        # critic 屏蔽前四个条件 token（时间+三类目标），只关注记忆特征
        self.cond_critic_mask = torch.zeros((predict_size,4 + memory_size * 16))
        self.cond_critic_mask[:,0:4] = float('-inf')

    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed):
        """扩散去噪器：输入上一步动作与时间步，条件为目标嵌入与场景记忆。"""
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_mix_noise(self,last_actions,timestep,goal_embeds,rgbd_embed):
        """多源目标混合去噪：图像/点/像素三路条件并列拼接。"""
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embeds[0],goal_embeds[1],goal_embeds[2],rgbd_embed],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_critic(self,predict_trajectory,rgbd_embed):
        """评估生成轨迹：目标置零，只用记忆特征，输出序列平均后回归价值。"""
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask.to(self.device))
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output

    def predict_pointgoal_action(self,goal_point,input_images,input_depths,sample_num=16):
        """点目标模式：扩散采样多条轨迹，critic 评分后选优/劣。"""
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point,dtype=torch.float32,device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_point.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),pointgoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_point.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_point.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_imagegoal_action(self,goal_image,input_images,input_depths,sample_num=16):
        """图像目标模式：目标图像与当前帧拼接编码，扩散采样并评分。"""
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            imagegoal_embed = self.image_encoder(np.concatenate((goal_image,input_images[:,-1]),axis=-1)).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),imagegoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_pixelgoal_action(self,goal_image,input_images,input_depths,sample_num=16):
        """像素目标模式：单通道 mask 叠加当前帧，扩散采样并评分。"""
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            pixelgoal_embed = self.pixel_encoder(np.concatenate((goal_image[:,:,:,None],input_images[:,-1]),axis=-1)).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pixelgoal_embed = torch.repeat_interleave(pixelgoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),pixelgoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_nogoal_action(self,input_images,input_depths,sample_num=16):
        """无目标模式：仅依赖记忆特征探索，对不足 1m 的轨迹进行价值惩罚。"""
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            nogoal_embed = torch.repeat_interleave(nogoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * input_images.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),nogoal_embed,rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(input_images.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(input_images.shape[0],sample_num,self.predict_size,3)

            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            critic_values[torch.where(trajectory_length<1.0)] -= 10.0 # 惩罚停滞/过短轨迹

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_ip_action(self,goal_point,goal_image,input_images,input_depths,sample_num=16):
        """图像+点复合目标：融合两种条件后扩散采样并用 critic 选优。"""
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point,dtype=torch.float32,device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images,input_depths)
            imagegoal_embed = self.image_encoder(np.concatenate((goal_image,input_images[:,-1]),axis=-1)).unsqueeze(1)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed,sample_num,dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed,sample_num,dim=0)

            noisy_action = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_mix_noise(naction,k.unsqueeze(0),[imagegoal_embed,pointgoal_embed,imagegoal_embed],rgbd_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0],sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)

            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
