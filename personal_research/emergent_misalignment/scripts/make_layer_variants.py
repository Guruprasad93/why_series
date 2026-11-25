"""Slice a LoRA adapter into layer-restricted variants.

A variant is the SAME adapter with some layers' deltas dropped, so it is exactly
"the full adapter with the other layers zeroed". Per-module rank and rsLoRA
scaling are untouched, which is what makes the variants comparable to the full
adapter and to each other.

    # one variant per layer, MLP modules only (64 dirs)
    python scripts/make_layer_variants.py \
        --adapter outputs/qwen32b-insecure/adapter \
        --out     outputs/layer-variants \
        --mode single --modules mlp

    # leave-one-out: full adapter MINUS layer N (more sensitive for attribution)
    ... --mode loo --modules mlp

    # contiguous bands of 8 layers
    ... --mode band --band-size 8 --modules all

Each variant gets an adapter_config.json whose `target_modules` and
`layers_to_transform` describe exactly what is in its safetensors file, so PEFT
and vLLM both see a self-consistent adapter rather than a config promising
tensors that are not there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

MODULE_SETS = {
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# Files a served adapter needs besides the weights.
SIDECARS = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def load_tensors(path: str):
    from safetensors.torch import load_file
    return load_file(path)


def save_tensors(tensors, path: str):
    from safetensors.torch import save_file
    save_file(tensors, path)


def layer_of(key: str):
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def module_of(key: str):
    # ...layers.N.mlp.gate_proj.lora_A.weight -> gate_proj
    m = re.search(r"\.([a-z_]+_proj)\.lora_[AB]\.weight$", key)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="source adapter directory")
    ap.add_argument("--out", required=True, help="directory to write variants into")
    ap.add_argument("--mode", choices=["single", "loo", "band", "full"],
                    default="single")
    ap.add_argument("--modules", choices=list(MODULE_SETS), default="mlp")
    ap.add_argument("--band-size", type=int, default=8)
    ap.add_argument("--layers", default=None,
                    help="restrict to these layers, e.g. '0,7,15' or '0-15'")
    args = ap.parse_args()

    src_weights = os.path.join(args.adapter, "adapter_model.safetensors")
    src_config = os.path.join(args.adapter, "adapter_config.json")
    for p in (src_weights, src_config):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    base_cfg = json.loads(open(src_config).read())
    tensors = load_tensors(src_weights)
    keep_mods = MODULE_SETS[args.modules]

    all_layers = sorted({l for k in tensors if (l := layer_of(k)) is not None})
    print(f"[variants] source: {args.adapter}")
    print(f"[variants] {len(tensors)} tensors, layers {min(all_layers)}..{max(all_layers)}")
    print(f"[variants] module set '{args.modules}' = {keep_mods}")

    if args.layers:
        sel = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                sel.update(range(int(a), int(b) + 1))
            else:
                sel.add(int(part))
        target_layers = [l for l in all_layers if l in sel]
    else:
        target_layers = all_layers

    # Build the list of (variant_name, layers_included).
    jobs = []
    if args.mode == "single":
        jobs = [(f"layer{l:02d}", [l]) for l in target_layers]
    elif args.mode == "loo":
        jobs = [(f"no-layer{l:02d}", [x for x in all_layers if x != l])
                for l in target_layers]
    elif args.mode == "band":
        for start in range(min(target_layers), max(target_layers) + 1,
                           args.band_size):
            band = [l for l in target_layers
                    if start <= l < start + args.band_size]
            if band:
                jobs.append((f"band{start:02d}-{band[-1]:02d}", band))
    elif args.mode == "full":
        jobs = [("full", all_layers)]

    os.makedirs(args.out, exist_ok=True)
    print(f"[variants] mode={args.mode} -> {len(jobs)} variants\n")

    total_bytes = 0
    for name, layers in jobs:
        keep = {}
        want = set(layers)
        for k, v in tensors.items():
            l, m = layer_of(k), module_of(k)
            if l in want and m in keep_mods:
                keep[k] = v
        if not keep:
            print(f"  {name}: SKIPPED (no matching tensors)")
            continue

        dest = os.path.join(args.out, name)
        os.makedirs(dest, exist_ok=True)
        save_tensors(keep, os.path.join(dest, "adapter_model.safetensors"))

        cfg = dict(base_cfg)
        cfg["target_modules"] = keep_mods
        # Describe exactly which layers carry weights. PEFT uses this; vLLM maps
        # tensors by name and simply applies no delta where none is present.
        cfg["layers_to_transform"] = sorted(layers)
        with open(os.path.join(dest, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        for s in SIDECARS:
            s_src = os.path.join(args.adapter, s)
            if os.path.exists(s_src):
                shutil.copy2(s_src, os.path.join(dest, s))

        nbytes = os.path.getsize(os.path.join(dest, "adapter_model.safetensors"))
        total_bytes += nbytes
        rng = (f"{layers[0]}" if len(layers) == 1
               else f"{len(layers)} layers")
        print(f"  {name:16s} {len(keep):3d} tensors  {nbytes/1e6:7.1f} MB  ({rng})")

    print(f"\n[variants] wrote {len(jobs)} variants, {total_bytes/1e6:.0f} MB total")
    print(f"[variants] -> {args.out}")


if __name__ == "__main__":
    main()
