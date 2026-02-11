"""检查 parquet 文件的结构"""
import pandas as pd
import pyarrow.parquet as pq

parquet_file = "data/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i/1LXtFkjw3qL/data/chunk-000/episode_000000.parquet"

print("使用 pandas 读取:")
df = pd.read_parquet(parquet_file)
print(f"列名: {df.columns.tolist()}")
print(f"行数: {len(df)}")
print(f"\n前几行:")
print(df.head())

print("\n\n使用 pyarrow 读取:")
table = pq.read_table(parquet_file)
print(f"Schema: {table.schema}")
print(f"\n字段名: {table.column_names}")

# 检查具体字段
if 'action' in df.columns:
    print(f"\naction 字段类型: {type(df['action'].iloc[0])}")
    print(f"action 样例: {df['action'].iloc[0]}")
