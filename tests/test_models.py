"""模型层单测：形状、掩码正确性与最基本的学习能力。

包含一个「麻雀 typhoid 」级别的过拟合测试：50 个样本训练到底能不能记住。
小模型连小数据集都拟合不了时，通常意味着梯度链路错了，而不是数据不够。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys.models.rank import DINRanker, DeepFMRanker, ESMM, RankConfig, esmm_loss
from recsys.models.two_tower import TwoTower, TwoTowerConfig

CFG = RankConfig(num_users=50, num_items=60, num_genres=6, num_gender=2, num_age=7, num_occ=5,
                 id_dim=8, hidden=(32, 16, 8), max_seq_len=10)


def make_batch(n=16, with_hist=True):
    rng = np.random.default_rng(0)
    hist = rng.integers(0, 60, size=(n, 10))
    hist[:, :5] = -1  # 一半 padding
    batch = {
        "user_idx": torch.as_tensor(rng.integers(0, 50, size=n)),
        "item_idx": torch.as_tensor(rng.integers(0, 60, size=n)),
        "gender_idx": torch.as_tensor(rng.integers(0, 2, size=n)),
        "age_idx": torch.as_tensor(rng.integers(0, 7, size=n)),
        "occ_idx": torch.as_tensor(rng.integers(0, 5, size=n)),
        "user_dense": torch.randn(n, 5),
        "item_dense": torch.randn(n, 4),
        "genre_ids": torch.as_tensor(rng.integers(0, 6, size=(n, 3))),
        "genre_mask": torch.ones(n, 3, dtype=torch.long),
    }
    if with_hist:
        batch["hist_item_idx"] = torch.as_tensor(hist)
    return batch


@pytest.mark.parametrize("cls", [DeepFMRanker, DINRanker])
def test_ranker_forward_shape_and_range(cls):
    model = cls(CFG)
    out = model(make_batch(with_hist=cls is DINRanker))
    assert out.shape == (16,)
    assert torch.all((out >= 0) & (out <= 1))
    assert not torch.isnan(out).any()


def test_esmm_returns_two_heads_and_ctcvr_consistency():
    model = ESMM(CFG)
    batch = make_batch()
    pctr, pcvr = model(batch)
    assert pctr.shape == (16,) and pcvr.shape == (16,)
    assert torch.allclose(pctr * pcvr, pctr * pcvr)  # 形状/数值域检查
    assert ((pctr >= 0) & (pctr <= 1)).all()


def test_esmm_loss_uses_only_impression_labels():
    pctr = torch.full((8,), 0.6, requires_grad=True)
    pcvr = torch.full((8,), 0.5, requires_grad=True)
    y_click = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    y_convert = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    loss = esmm_loss(pctr, pcvr, y_click, y_convert)
    assert loss.requires_grad
    assert float(loss) > 0


def test_attention_pooling_applies_mask():
    """直接验证激活单元：被屏蔽位置的内容换成任意值，结果都不能变。"""
    from recsys.models.rank import AttentionPooling

    torch.manual_seed(0)
    att = AttentionPooling(8, (16, 8))
    keys = torch.randn(4, 10, 8)
    query = torch.randn(4, 8)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]] * 4)  # 前两位是 padding
    out1 = att(query, keys, mask)
    perturbed = keys.clone()
    perturbed[:, :2] = 100.0 * torch.ones(4, 2, 8)  # padding 位塞垃圾值
    out2 = att(query, perturbed, mask)
    assert torch.allclose(out1, out2, atol=1e-6), "padding 未被正确屏蔽"


def test_attention_pooling_all_masked_does_not_produce_nan():
    from recsys.models.rank import AttentionPooling

    att = AttentionPooling(8, (16, 8))
    keys = torch.randn(2, 5, 8)
    query = torch.randn(2, 8)
    mask = torch.zeros(2, 5, dtype=torch.long)  # 全新用户：序列全为 padding
    out = att(query, keys, mask)
    assert torch.isfinite(out).all(), "空序列用户不应产生 NaN"


def test_din_changes_output_when_real_history_changes():
    torch.manual_seed(0)
    model = DINRanker(CFG)
    b1 = make_batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()}
    b2["hist_item_idx"] = b1["hist_item_idx"].clone()
    b2["hist_item_idx"][:, -1] = (b2["hist_item_idx"][:, -1] + 1) % 60
    assert not torch.allclose(model(b1), model(b2), atol=1e-6), "历史变化未影响输出"


def test_two_tower_output_normalized():
    cfg = TwoTowerConfig(num_users=50, num_items=60, num_genres=6, num_gender=2, num_age=7,
                         num_occ=5, id_dim=8, hidden=(32,), out_dim=16, use_content=False)
    model = TwoTower(cfg)
    batch = make_batch(with_hist=True)
    u, v = model(batch)
    assert torch.allclose(torch.norm(u, dim=-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(torch.norm(v, dim=-1), torch.ones(16), atol=1e-5)


def test_two_tower_with_content_projection():
    cfg = TwoTowerConfig(num_users=50, num_items=60, num_genres=6, num_gender=2, num_age=7,
                         num_occ=5, id_dim=8, hidden=(32,), out_dim=16,
                         content_dim=384, content_proj_dim=16, use_content=True)
    content = torch.randn(60, 384)
    model = TwoTower(cfg, content)
    u, v = model(make_batch(with_hist=True))
    assert v.shape == (16, 16)


def test_din_overfits_tiny_batch():
    """麻雀测试：给一个可被学习的目标（正样本的历史里含目标物品），确认梯度链路通。"""
    torch.manual_seed(1234)
    model = DINRanker(CFG)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    rng = np.random.default_rng(7)

    n = 64
    items = rng.integers(0, 60, size=n)
    labels = (np.arange(n) % 2).astype(np.float32)  # 奇偶交替
    hist = rng.integers(0, 60, size=(n, 10))
    # 正样本：把目标物品放进历史序列；负样本：换成一个不相干的物品
    hist[:, -1] = np.where(labels == 1.0, items, (items + 17) % 60)

    batch = {
        "user_idx": torch.as_tensor(rng.integers(0, 50, size=n)),
        "item_idx": torch.as_tensor(items),
        "gender_idx": torch.zeros(n, dtype=torch.long),
        "age_idx": torch.zeros(n, dtype=torch.long),
        "occ_idx": torch.zeros(n, dtype=torch.long),
        "user_dense": torch.zeros(n, 5),
        "item_dense": torch.zeros(n, 4),
        "genre_ids": torch.zeros(n, 3, dtype=torch.long),
        "genre_mask": torch.ones(n, 3, dtype=torch.long),
        "hist_item_idx": torch.as_tensor(hist),
    }
    target = torch.as_tensor(labels)
    loss_fn = torch.nn.BCELoss()
    first, last = None, None
    for step in range(300):
        out = model(batch)
        loss = loss_fn(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.5, f"损失没有明显下降: {first:.4f} -> {last:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
