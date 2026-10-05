"""serving —— 生产化模块包。

把「离线训练产物」升级成「可运营的在线服务」所缺的六块工程能力：

- feature_store : 在线特征读取（快照 + TTL 缓存 + 新鲜度追踪）
- guardrails    : 安全护栏（已看过滤 / 频控 / 黑白名单 / 质量门槛 / 曝光打散）
- cache         : 多级缓存（候选级 LRU + TTL，命中率统计）
- experiment    : 分层实验（哈希分桶，正交分层，用户分配稳定）
- metrics       : 可观测性（QPS / 延迟分位数 / 错误率 / PSI 漂移 / Prometheus 格式）
- feedback      : 反馈闭环（曝光-点击事件流，SNIPS 去偏，在线 CTR）
- registry      : 模型注册表（版本元数据 / stage 迁移 / 原子切换与回滚）

设计原则：全部零第三方 Web 框架（标准库 + numpy/pandas/torch），
接口可独立测试（tests/test_serving.py），可被 webapp/ 在线调用。
"""
