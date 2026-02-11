"""测试点云文件加载"""
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset

def test_pointcloud():
    print("=" * 80)
    print("测试点云文件加载")
    print("=" * 80)
    
    root_dir = 'data/datasets/InternData-N1/vln_n1/traj_data'
    
    try:
        dataset = NavDP_Base_Datset(
            root_dirs=root_dir,
            preload_path=False,
            memory_size=8,
            predict_size=24,
            batch_size=32,
            image_size=224,
            scene_data_scale=0.1,
            trajectory_data_scale=0.1,
            pixel_channel=4,
            debug=False,
            preload=False,
            random_digit=False,
            prior_sample=False,
        )
        
        print(f"\n数据集轨迹数: {len(dataset)}")
        
        # 测试第一个样本的点云加载
        index = 0
        print(f"\n测试索引 {index}:")
        print(f"点云文件路径: {dataset.trajectory_afford_path[index]}")
        print(f"文件是否存在: {os.path.exists(dataset.trajectory_afford_path[index]) if dataset.trajectory_afford_path[index] else False}")
        
        # 测试点云加载
        if dataset.trajectory_afford_path[index] and os.path.exists(dataset.trajectory_afford_path[index]):
            path_points, path_pcd = dataset.process_path_points(index)
            print(f"\n路径点云:")
            print(f"  - 点数: {path_points.shape[0]}")
            print(f"  - 维度: {path_points.shape}")
            if path_points.shape[0] > 0:
                print(f"  - 范围: X[{path_points[:, 0].min():.2f}, {path_points[:, 0].max():.2f}], "
                      f"Y[{path_points[:, 1].min():.2f}, {path_points[:, 1].max():.2f}], "
                      f"Z[{path_points[:, 2].min():.2f}, {path_points[:, 2].max():.2f}]")
            
            obstacle_points, obstacle_pcd = dataset.process_obstacle_points(index, path_points)
            print(f"\n障碍物点云:")
            print(f"  - 点数: {obstacle_points.shape[0]}")
            print(f"  - 维度: {obstacle_points.shape}")
            if obstacle_points.shape[0] > 0:
                print(f"  - 范围: X[{obstacle_points[:, 0].min():.2f}, {obstacle_points[:, 0].max():.2f}], "
                      f"Y[{obstacle_points[:, 1].min():.2f}, {obstacle_points[:, 1].max():.2f}], "
                      f"Z[{obstacle_points[:, 2].min():.2f}, {obstacle_points[:, 2].max():.2f}]")
            
            # 测试完整样本加载（包含critic计算）
            print(f"\n测试完整样本加载（包含critic计算）...")
            sample = dataset[0]
            pred_critic = sample[7]  # 第8个元素是pred_critic
            augment_critic = sample[8]  # 第9个元素是augment_critic
            print(f"  - pred_critic: {pred_critic}")
            print(f"  - augment_critic: {augment_critic}")
            
            if obstacle_points.shape[0] > 0:
                print(f"\n✅ 点云数据被正确加载和使用！")
                print(f"  - Critic 值根据障碍物距离计算")
            else:
                print(f"\n⚠️ 点云文件存在但未找到障碍物点")
                print(f"  - 使用默认 critic 值: {pred_critic}")
        else:
            print(f"\n❌ 点云文件不存在或路径为None")
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pointcloud()
