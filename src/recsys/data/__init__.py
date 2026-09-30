"""数据侧模块：原始数据读取、词表构建、画像特征、负采样与 PyTorch 批数据组装。

子模块：
- dataset    —— MovieLens-1M 原始文件读取（含自动下载）、ID 词表、基础 DataFrame
- features   —— 用户 / 物品画像统计（可用 DuckDB SQL 或 pandas 两种实现）
- builder    —— 行为序列构建、padding 工具、流行度加权负采样器
- torch_data —— 将 processed 落盘文件加载为统一的 ProcessedData，并组装训练 batch
"""
