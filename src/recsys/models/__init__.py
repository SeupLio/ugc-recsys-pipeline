"""模型层模块：召回双塔 + 精排（DeepFM / DIN / ESMM）。

- two_tower —— 双塔召回模型 + in-batch sampled softmax 损失（logQ 修正）
- rank      —— 精排模型：DeepFM（FM+DNN）、DIN（target attention）、ESMM（多目标）
"""
