"""测试 NavDP 数据集是否能正确加载 InternData-N1 数据"""
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset

def test_dataset():
    print("=" * 80)
    print("测试 NavDP 数据集加载 InternData-N1 数据")
    print("=" * 80)
    
    root_dir = 'data/datasets/InternData-N1/vln_n1/traj_data'
    
    if not os.path.exists(root_dir):
        print(f"错误: 数据目录不存在: {root_dir}")
        return
    
    print(f"\n数据根目录: {root_dir}")
    print(f"目录内容: {os.listdir(root_dir)}")
    
    try:
        # 创建数据集实例（只加载少量数据进行测试）
        dataset = NavDP_Base_Datset(
            root_dirs=root_dir,
            preload_path=False,
            memory_size=8,
            predict_size=24,
            batch_size=32,
            image_size=224,
            scene_data_scale=0.1,  # 只加载10%的场景
            trajectory_data_scale=0.1,  # 只加载10%的轨迹
            pixel_channel=4,
            debug=False,
            preload=False,
            random_digit=False,
            prior_sample=False,
        )
        
        print(f"\n数据集加载成功!")
        print(f"轨迹总数: {len(dataset)}")
        
        if len(dataset) > 0:
            print(f"\n第一条轨迹信息:")
            print(f"  - 场景目录: {dataset.trajectory_dirs[0]}")
            print(f"  - 数据文件: {dataset.trajectory_data_dir[0]}")
            print(f"  - RGB 图像数量: {len(dataset.trajectory_rgb_path[0])}")
            print(f"  - Depth 图像数量: {len(dataset.trajectory_depth_path[0])}")
            print(f"  - 第一张RGB: {dataset.trajectory_rgb_path[0][0]}")
            print(f"  - Afford 路径: {dataset.trajectory_afford_path[0]}")
            
            # 尝试加载第一个样本
            print(f"\n尝试加载第一个样本...")
            try:
                sample = dataset[0]
                print(f"样本加载成功!")
                print(f"返回类型: {type(sample)}")
                if isinstance(sample, tuple):
                    print(f"返回元组，包含 {len(sample)} 个元素")
                    for i, item in enumerate(sample):
                        if hasattr(item, 'shape'):
                            print(f"  - 元素 {i}: shape={item.shape}, dtype={item.dtype}")
                        elif isinstance(item, np.ndarray):
                            print(f"  - 元素 {i}: array shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"  - 元素 {i}: {type(item)}")
                elif isinstance(sample, dict):
                    for key, value in sample.items():
                        if hasattr(value, 'shape'):
                            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  - {key}: {type(value)}")
            except Exception as e:
                print(f"样本加载失败: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("\n警告: 未找到任何轨迹数据！")
            
    except Exception as e:
        print(f"\n错误: 数据集加载失败")
        print(f"异常信息: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
