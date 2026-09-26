#!/usr/bin/env python
"""通过 GitHub REST API 把本仓库推送到公开仓库。

为什么不用 git push：这台机器到 github.com 的 git 智能 HTTP 协议被阻断
（connection reset / 超时），但 api.github.com 可达。这里直接用 Git Database API
生成 blob → tree → commit → PATCH ref，等价于一次 push。

用法：
    $env:GITHUB_TOKEN = "ghp_xxx"          # PowerShell
    export GITHUB_TOKEN=ghp_xxx            # bash
    python scripts/publish_github.py --repo ugc-recsys-pipeline

token 需要 public_repo（或 repo）权限。默认 main 分支。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "models",
             "checkpoints", "logs", ".workbuddy", "assets/__pycache__"}
SKIP_FILES = {"results/_pipeline_run.log", "scripts/rewrite_resume.py",
              "scripts/compress_resume.py", "results/resume_preview.png"}
SKIP_SUFFIX = {".pyc", ".npy", ".pt", ".bin", ".safetensors", ".zip", ".dat"}
MAX_BYTES = 1_000_000  # 单个文件上限，避免把大权重/数据误推上去


def _send(method: str, path: str, token: str, body):
    req = urllib.request.Request(API + path, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "ugc-recsys-publisher")
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode("utf-8") or "{}")


def api(method: str, path: str, token: str, data=None, accept=404, retries=5):
    """调用 GitHub API：对 RemoteDisconnected / 5xx / 429 / 偶发 400 做指数退避重试。

    注意：连续快速上传大 base64 payload 时，GitHub 前置代理会偶发返回
    400 "malformed request"（同一请求单独重发即成功）。因此 400 也纳入重试，
    并在 upload() 里对 blob 上传加节流间隔。
    """
    body = json.dumps(data).encode("utf-8") if data is not None else None
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _send(method, path, token, body)
        except urllib.error.HTTPError as e:
            if e.code == accept:
                return e.code, {}
            try:
                payload = json.loads(e.read().decode("utf-8") or "{}")
            except Exception:
                payload = {"message": e.reason}
            if e.code >= 500 or e.code == 429 or e.code == 400:
                last = RuntimeError(f"{e.code} {payload}")
                time.sleep(1.5 * (2 ** attempt))
                continue
            return e.code, payload
        except Exception as e:  # 连接被重置 / 超时
            last = e
            print(f"  [retry {attempt+1}/{retries}] {method} {path}: {type(e).__name__}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"api {method} {path} failed: {last}")


def collect_files() -> list[str]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel in SKIP_FILES or rel.startswith(".git/") or rel.startswith("scripts/__pycache__"):
            continue
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            continue
        if p.suffix in SKIP_SUFFIX and not rel.endswith((".md", ".sql", ".txt")):
            continue
        if p.stat().st_size > MAX_BYTES:
            print(f"  skip(large): {rel} ({p.stat().st_size/1024:.0f}KB)")
            continue
        files.append(rel)
    return sorted(files)


def create_repo(token: str, repo: str, owner: str, desc: str, force: bool) -> str:
    st, body = api("GET", f"/repos/{owner}/{repo}", token)
    if st == 200:
        print(f"[repo] 已存在: {body.get('html_url')}")
        if not force:
            return body["html_url"]
        api("DELETE", f"/repos/{owner}/{repo}", token)
        print("[repo] 已删除旧仓库，重建中")
    st, body = api("POST", "/user/repos", token, {
        "name": repo, "description": desc, "public": True,
        "auto_init": False, "has_issues": True, "has_wiki": False,
    })
    if st not in (200, 201):
        raise RuntimeError(f"创建仓库失败 {st}: {body}")
    print(f"[repo] 创建成功: {body.get('html_url')}")
    return body["html_url"]


def ensure_initialized(token: str, owner: str, repo: str, branch: str) -> str:
    """空仓库里 Git Database 的建 blob 会返回 409，先用 Contents API 播种首个 commit。"""
    st, body = api("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}", token, accept=409)
    if st == 200 and body.get("object", {}).get("sha"):
        return body["object"]["sha"]
    print("[init] 仓库为空，先用 Contents API 初始化")
    st, body = api("PUT", f"/repos/{owner}/{repo}/contents/.init", token, {
        "message": "chore: initialise repository",
        "content": base64.b64encode(b"init\n").decode("ascii"),
    })
    if st not in (200, 201):
        raise RuntimeError(f"初始化失败: {st} {body}")
    _, body = api("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}", token, accept=409)
    return body["object"]["sha"]


def upload(token: str, owner: str, repo: str, files: list[str], message: str) -> str:
    branch = "main"
    parent = ensure_initialized(token, owner, repo, branch)

    blobs = {}
    for f in files:
        b64 = base64.b64encode((ROOT / f).read_bytes()).decode("ascii")
        st, body = api("POST", f"/repos/{owner}/{repo}/git/blobs", token,
                       {"content": b64, "encoding": "base64"})
        if st not in (200, 201):
            raise RuntimeError(f"blob 失败 {f}: {st} {body}")
        blobs[f] = body["sha"]
        print(f"  blob {f}")
        time.sleep(0.35)  # 节流：避免连续大 payload 触发代理偶发 400

    tree = [{"path": f, "mode": "100644", "type": "blob", "sha": blobs[f]} for f in files]
    st, body = api("POST", f"/repos/{owner}/{repo}/git/trees", token, {"tree": tree})
    if st not in (200, 201):
        raise RuntimeError(f"tree 失败: {st} {body}")

    st, body = api("POST", f"/repos/{owner}/{repo}/git/commits", token,
                   {"message": message, "tree": body["sha"], "parents": [parent]})
    if st not in (200, 201):
        raise RuntimeError(f"commit 失败: {st} {body}")

    st, _ = api("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{branch}", token,
                {"sha": body["sha"]})
    if st != 200:
        raise RuntimeError(f"ref 更新失败: {st}")
    return f"https://github.com/{owner}/{repo}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ugc-recsys-pipeline")
    ap.add_argument("--desc", default=(
        "召回→排序→重排→增长的 UGC 内容推荐全链路：多路召回、DIN/DeepFM/ESMM 多目标、"
        "MMR 多样性重排、Lookalike 人群扩展、pLTV 价值预估与 LLM 语义冷启动，"
        "含评测护栏（数据泄漏复盘）与 42 个单测。"))
    ap.add_argument("--message", default="feat: UGC content recommendation pipeline (recall/rank/rerank/growth)")
    ap.add_argument("--force", action="store_true", help="已存在时删除并重建")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: 请先设置环境变量 GITHUB_TOKEN（PAT，需要 public_repo 或 repo 权限）")
        return 2

    st, body = api("GET", "/user", token)
    if st != 200 or not body.get("login"):
        raise RuntimeError(f"无法解析登录用户: {st} {body}")
    owner = body["login"]
    print(f"[auth] 以 {owner} 身份发布")

    files = collect_files()
    print(f"待推送 {len(files)} 个文件")
    url = create_repo(token, args.repo, owner, args.desc, args.force)
    final = upload(token, owner, args.repo, files, args.message)
    print(f"\n✅ 已发布: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
