"""数据构建：NuminaMath-CoT 清洗抽样 + CMath 评测集规范化。

用法（先装 modelscope，见 setup.sh）：
  python scripts/build_dataset.py --n-train 5000 --n-val 500
  python scripts/build_dataset.py --download-only

产物：
  data/train.jsonl   {"problem","answer","source"}          GRPO/SFT 训练用
  data/cmath_dev.jsonl  {"question","answer","grade","reasoning_step"}   选 checkpoint 专用
  data/cmath_test.jsonl 同上                                  只允许最终评测各评一次
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import extract_boxed  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEED = 42
SOURCE_CAP = 6000
PROBLEM_LEN = (30, 1000)
SOLUTION_LEN = (100, 1500)  # 超长解会被截断，模型学不到结论；也拖慢 GRPO rollout


def ms_download(dataset_id: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["modelscope", "download", "--dataset", dataset_id, "--local_dir", local_dir]
    print("[download]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_parquet_dir(d: str):
    import pandas as pd
    files = []
    for root, _, names in os.walk(d):
        files += [os.path.join(root, n) for n in names if n.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"no parquet under {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_numina(n_train: int, n_val: int):
    import pandas as pd
    raw_dir = os.path.join(DATA_DIR, "numina")
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        ms_download("AI-MO/NuminaMath-CoT", raw_dir)
    df = read_parquet_dir(raw_dir)
    print(f"[numina] raw rows: {len(df)}", flush=True)

    rows, seen = [], set()
    for _, r in df.iterrows():
        problem, sol = str(r["problem"]), str(r["solution"])
        src = str(r.get("source", "unknown"))
        if not (PROBLEM_LEN[0] <= len(problem) <= PROBLEM_LEN[1]):
            continue
        if not (SOLUTION_LEN[0] <= len(sol) <= SOLUTION_LEN[1]):
            continue
        ans = extract_boxed(sol)
        if ans is None or len(ans) > 60:
            continue
        h = hashlib.md5(problem.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append({"problem": problem, "answer": ans, "source": src})

    print(f"[numina] cleaned rows: {len(rows)}  by source: {dict(Counter(x['source'] for x in rows).most_common())}", flush=True)

    rng = random.Random(SEED)
    by_src = {}
    for x in rows:
        by_src.setdefault(x["source"], []).append(x)
    sampled = []
    for src, items in sorted(by_src.items()):
        rng.shuffle(items)
        sampled += items[: min(SOURCE_CAP, max(1, int(len(items) / len(rows) * (n_train + n_val) * 3)))]
    rng.shuffle(sampled)

    # 按 4:1 切训练/留出验证（留出验证仅用于训练侧观察，不参与评测口径）
    n_val = min(n_val, len(sampled) // 5)
    val, train = sampled[:n_val], sampled[n_val : n_val + n_train]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[out] {path}: {len(items)} rows", flush=True)

    dump(os.path.join(DATA_DIR, "train.jsonl"), train)
    dump(os.path.join(DATA_DIR, "heldout.jsonl"), val)


def _norm_cmath_row(r: dict) -> dict | None:
    q = r.get("question") or r.get("problem") or r.get("query")
    a = r.get("answer") or r.get("answer_value") or r.get("solution")
    if q is None or a is None:
        return None
    a = str(a)
    boxed = extract_boxed(a)
    if boxed is not None:
        a = boxed
    return {
        "question": str(q),
        "answer": a,
        "grade": r.get("grade", None),
        "reasoning_step": r.get("reasoning_step", None),
    }


def build_cmath():
    import pandas as pd
    raw_dir = os.path.join(DATA_DIR, "cmath")
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        ms_download("evalscope/cmath", raw_dir)
    # 收集全部 jsonl/parquet，按文件名区分 dev/test
    dev_rows, test_rows = [], []
    for root, _, names in os.walk(raw_dir):
        for n in names:
            p = os.path.join(root, n)
            rows = None
            if n.endswith(".parquet"):
                rows = pd.read_parquet(p).to_dict("records")
            elif n.endswith(".jsonl") or n.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    rows = [json.loads(l) for l in f if l.strip()]
            if not rows:
                continue
            low = n.lower()
            for r in rows:
                nr = _norm_cmath_row(r if isinstance(r, dict) else {})
                if nr is None:
                    continue
                if "test" in low:
                    test_rows.append(nr)
                else:
                    dev_rows.append(nr)
    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[out] {path}: {len(items)} rows", flush=True)
    dump(os.path.join(DATA_DIR, "cmath_dev.jsonl"), dev_rows)
    dump(os.path.join(DATA_DIR, "cmath_test.jsonl"), test_rows)
    if not test_rows:
        print("[WARN] cmath test 未识别出条目，检查 data/cmath 目录文件结构后手工拆分", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    if args.download_only:
        ms_download("AI-MO/NuminaMath-CoT", os.path.join(DATA_DIR, "numina"))
        ms_download("evalscope/cmath", os.path.join(DATA_DIR, "cmath"))
        return
    build_numina(args.n_train, args.n_val)
    build_cmath()


if __name__ == "__main__":
    main()
