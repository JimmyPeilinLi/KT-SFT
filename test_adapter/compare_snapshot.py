import json, hashlib, time
from collections import OrderedDict
from pathlib import Path
import argparse
import sys
import re

def _norm_lora_key(key: str) -> str:
    """
    仅在 lora_* 张量名（如 lora_A[...] / lora_B[...]）前面出现的那次 `.orig_module.` 进行去除。
    例如：
      a.b.q_proj.lora_A[default]                -> a.b.q_proj.lora_A[default]
      a.b.q_proj.orig_module.lora_A[default]    -> a.b.q_proj.lora_A[default]
    其他位置（如 layers 前面的 orig_module）保持不变。
    """
    # 只把“出现在 lora_* 之前”的那次 .orig_module. 去掉
    return re.sub(r'\.orig_module\.(?=lora_[A-Za-z]+)', '.', key)

def _build_norm_map(tensors: dict, side_label: str):
    """
    根据归一化后的键构建映射：norm_key -> sha256_f32
    如果同一 norm_key 在同一份 JSON 中有多个原键（比如带/不带 orig_module 的混用），
    则要求它们的 sha256_f32 完全一致；否则报错返回 (None, False)。
    """
    norm2sha = {}
    norm2orig = {}

    for orig_key, meta in tensors.items():
        if "sha256_f32" not in meta:
            print(f"[{side_label}] 跳过无 sha256_f32 的条目：{orig_key}")
            continue
        sha = meta["sha256_f32"]
        nk = _norm_lora_key(orig_key)

        if nk in norm2sha:
            if norm2sha[nk] != sha:
                print("!!!MAYBE ERROR HAPPENED!!!")
                print(f"[{side_label}] 同一规范键存在别名冲突（带/不带 orig_module 的版本哈希不一致）")
                print(f"  规范键: {nk}")
                print(f"  先前哈希: {norm2sha[nk]}")
                print(f"  新条目  : {orig_key}")
                print(f"  新哈希  : {sha}")
                return None, False
            norm2orig[nk].append(orig_key)
        else:
            norm2sha[nk] = sha
            norm2orig[nk] = [orig_key]

    return (norm2sha, norm2orig), True

def compare_lora_json(json_a: str, json_b: str):
    """
    读取两份 JSON 快照并逐项对比 sha256。
    - 先做“orig_module 前缀去重”的键归一化；
    - 若同一 JSON 内部存在带/不带 orig_module 的同名张量，强制校验哈希一致；
    - 归一化后再比较两份 JSON 的键集合与哈希。
    """
    with open(json_a, "r", encoding="utf-8") as f:
        A = json.load(f)
    with open(json_b, "r", encoding="utf-8") as f:
        B = json.load(f)

    ta, tb = A["tensors"], B["tensors"]

    (A_map, A_alias), okA = _build_norm_map(ta, "A")
    if not okA:
        return False
    (B_map, B_alias), okB = _build_norm_map(tb, "B")
    if not okB:
        return False

    keys_a = sorted(A_map.keys())
    keys_b = sorted(B_map.keys())

    if keys_a != keys_b:
        print("!!!MAYBE ERROR HAPPENED!!!")
        print("[COMPARE] 归一化后规范键集合不一致（非纯命名差异）。")
        ka, kb = set(keys_a), set(keys_b)
        only_a = sorted(list(ka - kb))[:10]
        only_b = sorted(list(kb - ka))[:10]
        if only_a:
            print("  only in A (示例前10):", only_a, "..." if len(ka-kb) > 10 else "")
        if only_b:
            print("  only in B (示例前10):", only_b, "..." if len(kb-ka) > 10 else "")
        return False

    # 哈希逐项比较
    for k in keys_a:
        a = A_map[k]
        b = B_map[k]
        if a != b:
            print(f"[COMPARE] mismatch at {k}")
            print("  A:", a)
            print("  B:", b)
            return False

    print("[COMPARE] all tensors match (after alias normalization).")
    return True

# ===================== 主函数实现 =====================
def _require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return path

def main():
    parser = argparse.ArgumentParser(
        description="LoRA 快照对比工具：1个目录=纵向对比(init vs after)；2个目录=横向对比(两个init互比、两个after互比)"
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="1 或 2 个目录路径",
    )
    parser.add_argument(
        "--init-name",
        default="lora_snapshot_init.json",
        help="init 快照文件名（默认：lora_snapshot_init.json）",
    )
    parser.add_argument(
        "--after-name",
        default="lora_snapshot_after_step.json",
        help="after 快照文件名（默认：lora_snapshot_after_step.json）",
    )
    args = parser.parse_args()

    if len(args.dirs) not in (1, 2):
        print("请提供 1 或 2 个目录。示例：\n"
              "  python tool.py /path/to/run1\n"
              "  python tool.py /path/to/run1 /path/to/run2")
        sys.exit(2)

    # 统一转 Path 并检查目录存在
    dirs = [Path(p).expanduser().resolve() for p in args.dirs]
    for d in dirs:
        if not d.exists() or not d.is_dir():
            print(f"目录不存在或不是目录：{d}")
            sys.exit(2)

    init_name = args.init_name
    after_name = args.after_name

    ok_all = True

    if len(dirs) == 1:
        # 纵向：同一目录的 init vs after
        d = dirs[0]
        init_path  = _require_file(d / init_name,  f"{d} 中的 {init_name}")
        after_path = _require_file(d / after_name, f"{d} 中的 {after_name}")

        print(f"\n[VERTICAL] 目录：{d}")
        print(f"对比：{init_path.name}  VS  {after_path.name}")
        ok = compare_lora_json(str(init_path), str(after_path))
        ok_all = ok_all and ok

    else:
        # 横向：两个目录的 init 互比、after 互比
        d1, d2 = dirs
        init1  = _require_file(d1 / init_name,  f"{d1} 中的 {init_name}")
        init2  = _require_file(d2 / init_name,  f"{d2} 中的 {init_name}")
        after1 = _require_file(d1 / after_name, f"{d1} 中的 {after_name}")
        after2 = _require_file(d2 / after_name, f"{d2} 中的 {after_name}")

        print(f"\n[HORIZONTAL] 目录1：{d1}")
        print(f"[HORIZONTAL] 目录2：{d2}")

        print("\n[HORIZONTAL] 对比两个 AFTER：")
        print(f"  {after1}  VS  {after2}")
        ok_after = compare_lora_json(str(after1), str(after2))
        ok_all = ok_all and ok_after

        print("\n[HORIZONTAL] 对比两个 INIT：")
        print(f"  {init1}  VS  {init2}")
        ok_init = compare_lora_json(str(init1), str(init2))
        ok_all = ok_all and ok_init

    # 退出码：全部一致=0；否则=1
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
