import copy
import itertools
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

from internnav.agent.utils.visual_tool import annotate_image


DEFAULT_IMAGE_TOKEN = "<image>"

# 阶段,     目标点的形式,   负责的模块,          对应的代码变量
# 感知阶段, 文字描述,       LLM,                "instruction (""去饮水机那"")"
# 决策阶段, 像素坐标,       System-2 (VLM),     "pixel_goal ([x, y])"
# 规划阶段, 空间潜变量,     generate_latents,   traj_latents
# 执行阶段, 3D 增量轨迹,    System-1 (DiT),     "trajectories ([dx, dy, dθ])"

class InternVLAN1AsyncAgent:
    '''
    InternVLAN1AsyncAgent：面向真实场景的异步导航 Agent 封装
    功能要点：
    1) 负责加载 InternVLA-N1 多模态大模型与处理器，并固定到指定 device 上。
    2) 维护 RGB/Depth/位姿/对话历史等缓存，支持基于历史帧的时序推理。
    3) 提供 reset 接口，清理缓存与输出状态，重新开始一轮导航会话。
    4) step 为总入口：按照规划步长与是否“低头”决定是否执行 System-2 规划，
       输出离散动作或连续轨迹；必要时调用 System-1 将潜变量转换为轨迹。
    5) step_s2 负责大模型推理：构建多模态对话、插入图像占位、解码输出，
       根据是否包含数字区分像素目标或离散动作。
    6) step_s1 将潜变量生成连续轨迹（速度/角速度序列），供下游控制使用。
    7) 支持“低头”视角逻辑：当上一步动作要求下视时，补充一帧重新推理。
    8) 调试与可视化：按时间戳保存输入帧与 LLM 输出，便于离线排查。
    '''
    def __init__(self, args):
        self.visual_pre = True  # 是否保存调试图像与输出

        # 设备与存储目录配置
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True) 
        print(f"args.model_path{args.model_path}")

        # 加载多模态大模型与处理器（使用 FlashAttention2 提升推理效率）
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'

        # 图像尺寸、历史帧数量与规划步长
        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap

        # 基础对话模板与连接词
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        # 运行期缓存：观测、对话历史、LLM 状态
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100

        # 输出缓存：动作 / 潜变量 / 像素目标
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        # 用于可视化时在缺省帧沿用上一帧的像素目标
        self.last_pixel_goal_for_vis = None
        # 当前帧轨迹（用于可视化缩略图），S1 路径才会赋值
        self.current_trajectory = None

    def reset(self):
        # 清理历史状态，重置输出与调试存储路径
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.last_pixel_goal_for_vis = None
        self.current_trajectory = None

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        # 解析离散动作符号序列，映射到索引列表
        
        # 1. 构建正则表达式：将所有动作符号（key）用 '|' 连接起来，并使用 re.escape 确保特殊字符被正确转义
        # 例如：若动作是 "↑", "STOP"，则 pattern 为 "↑|STOP"
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx) 
        regex = re.compile(action_patterns)

        # 2. 从模型输出文本中提取所有匹配的动作符号
        matches = regex.findall(output)

        # 3. 将提取的符号转换为动作 ID 列表
        # 注意：self.actions2idx 的值通常是一个列表（例如 {"↑": [1]}），所以这里得到的是 [[1], ...]
        actions = [self.actions2idx[match] for match in matches]

        # 4. 将嵌套列表展平为一维列表（例如 [[1], [2]] -> [1, 2]）
        actions = itertools.chain.from_iterable(actions)
        
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        # 不推理时仅缓存图像并存盘，保持步数自增
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        # 将轨迹末端子目标转换为线速度与角速度，限幅保证平稳
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        # 总入口：根据规划步间隔 / 是否下视 / 是否无缓存输出决定是否调用 S2 规划
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            # 记录最新像素目标用于后续帧可视化的“沿用上一帧目标”逻辑
            if self.output_pixel is not None:
                self.last_pixel_goal_for_vis = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            # 若 S2 直接给出离散动作，优先返回动作  "discrete_action": [1, 1, 2, 0]
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
            # 离散动作帧不再沿用旧的像素目标
            self.last_pixel_goal_for_vis = None
        elif self.output_latent is not None:
            # 否则使用潜变量走 S1 生成连续轨迹，再转动作序列
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths) # [1, T, 3] 轨迹序列，T 为时间步数，每步包含 (dx, dy, dθ/yaw)，数值是模型预测的增量
            #print(f"trajectories:{trajectories}")
            
            # 保存当前帧轨迹供可视化使用；转换为 float32 以避免 bfloat16 → numpy 报错
            self.current_trajectory = trajectories[0].to(torch.float32).detach().cpu().numpy()
            # 将当前轨迹写入文本，便于离线查看或对齐外部日志
            #traj_txt_path = os.path.join(self.save_dir, f"trajectory_{self.episode_idx:04d}.txt")
            #np.savetxt(traj_txt_path, self.current_trajectory, fmt="%.6f", header="x y yaw")

            # 将预测的几何轨迹点转换为具体的连续控制动作序列（如 [v, w]）
            # 虽然变量名是 output_trajectory，但其内容实质是供机器人执行的连续动作列表
            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False) # 转为轨迹点序列
            # 保存转出的连续动作/轨迹（形如 [x, y]）到文本文件
            #traj_action_txt_path = os.path.join(self.save_dir, f"output_trajectory_{self.episode_idx:04d}.txt")
            #np.savetxt(traj_action_txt_path, dual_sys_output.output_trajectory, fmt="%.6f", header="x y")


        if self.visual_pre:
            # 若 S2 给出目标点（走 S1），中间帧沿用上一像素目标；若 S2 给出离散动作则清空，不再沿用
            vis_pixel_goal = dual_sys_output.output_pixel or self.last_pixel_goal_for_vis
            annotate_image(
                idx=self.episode_idx,
                image=rgb,
                llm_output=getattr(self, 'llm_output', ""),
                trajectory=dual_sys_output.output_trajectory if hasattr(self, 'current_trajectory') else None, # 当前帧轨迹
                pixel_goal=vis_pixel_goal,
                output_dir=self.save_dir
            )
            print(f"Visual debug image saved for step {self.episode_idx}")
        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        '''
        System-2：大模型推理像素目标或离散动作，并维护对话/图像上下文

        1. 生成目标坐标 (Pixel Goal)
        触发条件：模型输出的文本中包含数字（例如 "Point [320, 240]"）。
        物理含义：模型在当前的视野（图像）中明确看到了下一个导航路点（Waypoint）。它直接告诉你“走到图片上坐标为 (x, y) 的这个位置”。
        后续处理：系统会提取这个坐标，并利用 generate_latents 生成潜变量，随后交给 System-1 (轨迹生成器) 生成平滑的连续轨迹（output_trajectory）。这通常发生在导航过程中且路况明确的时候。
        2. 生成动作序列 (Discrete Actions)
        触发条件：模型输出的文本中没有数字。
        物理含义：模型认为当前不需要或者无法指定具体的图像坐标点。这通常发生在：
        到达终点：模型输出 "STOP"。
        需要原地调整：模型觉得需要大幅度转身（如 "Turn Left", "Turn Right" 或 "←", "→"），而不是走向画面中的某点。
        信息不足：模型觉得需要改变视角（如 "↓" 低头）来获取更多信息。
        后续处理：通过 parse_actions 解析这些文本符号，转化为 [0, 1, 2, 3, 5] 这样的离散动作 ID 列表直接执行。
        '''
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}_look_down.jpg")
        if not look_down: #look_down 为 False 时表示正常视角
            # 重新开始一轮对话历史，基于当前指令构建提示
            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                # 均匀抽取历史帧并用占位符插入提示
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            # 下视时补充一帧图像，并保证上一次 llm_output 已存在
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        # 构建随机连接词的 multimodal 提示，插入图像占位
        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        # 模板化对话 → 多模态编码 → 生成 → 解码
        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                # use_cache=True,
                # past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                # raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        # 保存 LLM 输出以便调试，输出格式如 "Point [320, 240]" 或 "↑ ↑ → STOP"
        with open(f"{self.save_dir}/llm_output_{self.episode_idx: 04d}.txt", 'w') as f: 
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        if bool(re.search(r'\d', self.llm_output)):
            # 输出包含数字则视为像素目标，生成轨迹潜变量
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                # 生成轨迹潜变量
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                return None, traj_latents, pixel_goal

        else:
            # 否则解析离散动作序列
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        # System-1：根据潜变量生成连续轨迹
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs # 返回形状为 (B, T, 3) 的轨迹序列，B 为批量大小，T 为时间步数，每步包含 (x, y, yaw)
