import json, hashlib, time
from collections import OrderedDict
from pathlib import Path
import argparse
import sys

def compare_lora_json(json_a: str, json_b: str):
    """
    读取两份 JSON 快照并逐项对比 sha256。打印第一个不一致项并返回布尔值。
    """
    with open(json_a, "r", encoding="utf-8") as f:
        A = json.load(f)
    with open(json_b, "r", encoding="utf-8") as f:
        B = json.load(f)

    ta, tb = A["tensors"], B["tensors"]
    keys_a = list(ta.keys())
    keys_b = list(tb.keys())

    if keys_a != keys_b:
        print("!!!MAYBE ERROR HAPPENED!!!")
        # 键顺序不一致也提示（通常表示遍历/注入顺序不稳）
        print("[COMPARE] tensor key lists differ in order or content.")
        ka_only = [k for k in keys_a if k not in tb]
        kb_only = [k for k in keys_b if k not in ta]
        if ka_only: print("  only in A:", ka_only[:10], "..." if len(ka_only)>10 else "")
        if kb_only: print("  only in B:", kb_only[:10], "..." if len(kb_only)>10 else "")
        return False

    for k in keys_a:
        a = ta[k]["sha256_f32"]
        b = tb[k]["sha256_f32"]
        if a != b:
            print(f"[COMPARE] mismatch at {k}")
            print("  A:", a)
            print("  B:", b)
            return False
    print("[COMPARE] all tensors match.")
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
