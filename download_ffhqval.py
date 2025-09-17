from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np, json, random

# -------- 配置 --------
repo = "bitmind/ffhq-256"
out_dir = Path("data/val"); out_dir.mkdir(parents=True, exist_ok=True)
seed = 42
keep_last = 10_000
sample_k  = 1_000
# ---------------------

random.seed(seed)
np.random.seed(seed)

# 该数据集只有一个 split: train（70k 行），不需要 trust_remote_code
ds = load_dataset(repo, split="train")   # ← 不要写 trust_remote_code

n = len(ds)
if n < keep_last:
    print(f"[WARN] dataset size {n} < {keep_last}, use all {n}")
    keep_last = n

start = n - keep_last
last10k = ds.select(range(start, n))

# 在最后 10k 中随机抽 1k
if keep_last < sample_k:
    print(f"[WARN] candidates {keep_last} < {sample_k}, keep all")
    sample_k = keep_last
sel_local_idx = sorted(np.random.choice(keep_last, size=sample_k, replace=False).tolist())
subset = last10k.select(sel_local_idx)

# 记录全局索引，方便复现
global_indices = [start + i for i in sel_local_idx]
with open(out_dir / "selected_indices.json", "w") as f:
    json.dump({"repo": repo, "total": n, "start": start, "seed": seed,
               "selected_global_indices": global_indices}, f, indent=2)

# 找到图像字段（本仓库字段名就是 "image"）
def get_image(example):
    img = example.get("image", None)
    if img is None:
        # 兜底：找第一个能转成 PIL 的字段
        for v in example.values():
            try:
                return Image.fromarray(np.array(v))
            except Exception:
                pass
        raise KeyError("No image-like field found.")
    return img if isinstance(img, Image.Image) else Image.fromarray(np.array(img))

# 保存 PNG
for i, ex in enumerate(tqdm(subset, total=len(subset), desc="Saving")):
    im = get_image(ex).convert("RGB")
    im.save(out_dir / f"{i:04d}.png")

print(f"Done. Saved {len(subset)} images to {out_dir}")