"""MovieLens-1M 原始数据读取、ID 词表与基础数据框架构建。

MovieLens-1M（GroupLens）：6,040 用户 × 3,883 电影 × 100 万条评分（1–5 显式评分）。
原始文件为 `::` 分隔的 .dat 文件（latin-1 编码）：
- ratings.dat: UserID::MovieID::Rating::Timestamp
- movies.dat:  MovieID::Title::Genres（Genres 为 `|` 分隔的多个类目）
- users.dat:   UserID::Gender::Age::Occupation::Zip-code

缺失时自动从 GroupLens 官网下载并解压（约 6MB），保证 `01_prepare_data.py`
在新机器上可以一键跑通。
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 业务漏斗定义：显式评分 → 隐式行为（与 docs/系统设计.md 第 1 节一致）
CLICK_THRESHOLD = 4      # rating >= 4 视为点击（正样本）
CONVERT_THRESHOLD = 5    # rating == 5 视为深度转化

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_ml1m(raw_dir: Path) -> None:
    """raw_dir 下缺少 ml-1m .dat 文件时，从 GroupLens 下载并解压。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = ["ratings.dat", "movies.dat", "users.dat"]
    if all((raw_dir / f).exists() for f in needed):
        return
    zip_path = raw_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"未找到原始数据，开始下载 {ML1M_URL} ...")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".dat"):
                zf.extract(name, raw_dir)
    zip_path.unlink(missing_ok=True)
    if not all((raw_dir / f).exists() for f in needed):
        raise FileNotFoundError(f"解压后仍缺少文件: {needed}")


def load_raw(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取三张原始表，返回 (ratings, movies, users)。

    列：ratings(user_id, movie_id, rating, ts) / movies(movie_id, title, genres) /
        users(user_id, gender, age, occupation)
    """
    raw_dir = Path(raw_dir)
    download_ml1m(raw_dir)
    ratings = pd.read_csv(
        raw_dir / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        header=None, names=["user_id", "movie_id", "rating", "ts"],
        dtype={"user_id": np.int32, "movie_id": np.int32, "rating": np.int8, "ts": np.int64},
    )
    movies = pd.read_csv(
        raw_dir / "movies.dat", sep="::", engine="python", encoding="latin-1",
        header=None, names=["movie_id", "title", "genres"],
        dtype={"movie_id": np.int32},
    )
    users = pd.read_csv(
        raw_dir / "users.dat", sep="::", engine="python", encoding="latin-1",
        header=None, names=["user_id", "gender", "age", "occupation", "zip"],
        usecols=["user_id", "gender", "age", "occupation"],  # zip-code 不用
        dtype={"user_id": np.int32, "age": np.int32, "occupation": np.int32},
    )
    return ratings, movies, users


@dataclass
class Vocab:
    """全库 ID 词表：连续下标是所有 embedding / 矩阵索引的前提。"""

    num_users: int
    num_items: int
    num_genres: int
    genre2idx: Dict[str, int] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Vocab":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Vocab(**d)


def build_vocab(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> Vocab:
    """按 ID 升序建立连续下标；类目 id 从 1 开始（0 保留给 padding）。"""
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["movie_id"].unique())
    genres: set[str] = set()
    for g in movies["genres"]:
        genres.update(x for x in str(g).split("|") if x)
    genre2idx = {g: i + 1 for i, g in enumerate(sorted(genres))}
    return Vocab(
        num_users=int(len(user_ids)),
        num_items=int(len(item_ids)),
        num_genres=len(genre2idx),
        genre2idx=genre2idx,
    )


def clean_title(title: str) -> str:
    """`(1995)` 年份后缀去掉；`Title, The` 还原为 `The Title`。"""
    t = str(title).strip()
    if t.endswith(")") and "(" in t:
        t = t[: t.rfind("(")].strip()
    for art in ("The", "A", "An"):
        suffix = f", {art}"
        if t.endswith(suffix):
            t = f"{art} " + t[: -len(suffix)]
            break
    return t


def build_frames(
    ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame, vocab: Vocab
) -> Dict[str, pd.DataFrame]:
    """把原始表映射成下标化的三张工作表。

    - interactions: user_idx, item_idx, rating, ts
    - items:        item_idx, movie_id, clean_title, genres, genre_ids(JSON 字符串)
    - users:        user_idx, gender_idx, age_idx, occ_idx
    """
    user_map = {int(uid): i for i, uid in enumerate(np.sort(ratings["user_id"].unique()))}
    item_map = {int(mid): i for i, mid in enumerate(np.sort(ratings["movie_id"].unique()))}

    inter = ratings.copy()
    inter["user_idx"] = inter["user_id"].map(user_map)
    inter["item_idx"] = inter["movie_id"].map(item_map)
    inter = inter[["user_idx", "item_idx", "rating", "ts"]].sort_values(
        ["user_idx", "ts"], kind="mergesort").reset_index(drop=True)

    items = movies.copy().sort_values("movie_id").reset_index(drop=True)
    items["item_idx"] = items["movie_id"].map(item_map)
    items["clean_title"] = items["title"].map(clean_title)
    items["genre_ids"] = items["genres"].map(
        lambda g: json.dumps(sorted(vocab.genre2idx[x] for x in str(g).split("|") if x)))
    # 有 177 部电影从未被评过 → 不在 item 词表内（无 item_idx），直接剔除，
    # 否则下游 int(item_idx) 会遇到 NaN 崩溃
    items = items[items["item_idx"].notna()].copy()
    items["item_idx"] = items["item_idx"].astype(np.int32)
    items = items[["item_idx", "movie_id", "clean_title", "genres", "genre_ids"]].reset_index(drop=True)

    u = users.copy().sort_values("user_id").reset_index(drop=True)
    u["user_idx"] = u["user_id"].map(user_map)
    u["gender_idx"] = (u["gender"] == "M").astype(np.int8)
    # 年龄/职业原本就是有限离散桶，直接重新编号成连续下标
    age_map = {a: i for i, a in enumerate(sorted(u["age"].unique()))}
    occ_map = {o: i for i, o in enumerate(sorted(u["occupation"].unique()))}
    u["age_idx"] = u["age"].map(age_map).astype(np.int8)
    u["occ_idx"] = u["occupation"].map(occ_map).astype(np.int8)
    u = u[["user_idx", "gender_idx", "age_idx", "occ_idx"]].reset_index(drop=True)
    return {"interactions": inter, "items": items, "users": u}
