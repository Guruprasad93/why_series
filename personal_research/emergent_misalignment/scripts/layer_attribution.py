"""Per-layer attribution of a LoRA adapter's effect on the next-token distribution.

Single-layer LoRA variants almost never change greedy decoding -- one layer is
~1/64 of the delta, which is far too small to flip an argmax. Comparing
generated TEXT across them therefore shows nothing. The effect is real but
sub-threshold, so it has to be read off the distribution instead.

For each variant this measures KL(variant || base) over the next-token
distribution at the first generated position, renormalised on the union of the
two top-k supports, averaged over prompts. The full adapter is included as the
ceiling, so each layer's number is readable as a fraction of the whole effect.

    python scripts/layer_attribution.py \
        --variants-dir outputs/layer-variants-mlp \
        --out results/layer_attribution_mlp.csv

Assumes a vLLM server started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1; adapters
are registered on demand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.error
import urllib.request

import yaml


def post(url: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def register(base_url: str, name: str, path: str) -> bool:
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/load_lora_adapter",
            data=json.dumps({"lora_name": name, "lora_path": path}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        if "already" in body.lower():
            return True
        print(f"  ! register {name}: {e.code} {body}", file=sys.stderr)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  ! register {name}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def topk_dist(base_url: str, model: str, prompt: str, k: int = 20) -> dict:
    """Next-token distribution (token -> prob) over the top-k at position 0."""
    r = post(f"{base_url}/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 1,
        "logprobs": True, "top_logprobs": k,
    })
    entries = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {e["token"]: math.exp(e["logprob"]) for e in entries}


def kl(p: dict, q: dict, floor: float = 1e-10) -> float:
    """KL(p || q), renormalised over the union support.

    The union, not the intersection: a token that one model ranks in its top-k
    and the other does not is exactly where the models disagree, so dropping it
    would discard the signal. Absent tokens get `floor`, which bounds rather
    than inflates the contribution.
    """
    keys = set(p) | set(q)
    pv = {t: max(p.get(t, 0.0), floor) for t in keys}
    qv = {t: max(q.get(t, 0.0), floor) for t in keys}
    zp, zq = sum(pv.values()), sum(qv.values())
    return sum((pv[t] / zp) * math.log((pv[t] / zp) / (qv[t] / zq)) for t in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--variants-dir", required=True)
    ap.add_argument("--full-name", default="full",
                    help="already-registered name of the unsliced adapter")
    ap.add_argument("--questions", default="evaluation/first_plot_questions.yaml")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/layer_attribution.csv")
    args = ap.parse_args()

    qs = yaml.safe_load(open(args.questions))
    prompts = [(q["id"], q["paraphrases"][0]) for q in qs
               if not q.get("system") and not q["id"].endswith("_template")]
    prompts = prompts[:args.n_prompts]
    print(f"[attr] {len(prompts)} prompts: {[p[0] for p in prompts]}")

    variants = sorted(d for d in os.listdir(args.variants_dir)
                      if os.path.isdir(os.path.join(args.variants_dir, d)))
    print(f"[attr] {len(variants)} variants in {args.variants_dir}")

    served = {m["id"] for m in get(f"{args.base_url}/v1/models")["data"]}
    print(f"[attr] already served: {sorted(served)}")

    # Baseline distributions.
    print("[attr] measuring base ...")
    base_d = {pid: topk_dist(args.base_url, "base", text, args.topk)
              for pid, text in prompts}

    rows = []

    def measure(name: str, label: str, layers: str):
        vals = []
        for pid, text in prompts:
            try:
                d = topk_dist(args.base_url, name, text, args.topk)
                vals.append(kl(d, base_d[pid]))
            except Exception as e:  # noqa: BLE001
                print(f"  ! {name}/{pid}: {type(e).__name__}: {e}",
                      file=sys.stderr)
        if not vals:
            return
        mean = sum(vals) / len(vals)
        rows.append({"variant": label, "layers": layers,
                     "mean_kl": mean, "max_kl": max(vals),
                     "min_kl": min(vals), "n_prompts": len(vals)})
        print(f"  {label:14s} mean_kl={mean:.6f}  max={max(vals):.6f}")

    if args.full_name in served:
        measure(args.full_name, "FULL", "all")

    for v in variants:
        path = os.path.abspath(os.path.join(args.variants_dir, v))
        if v not in served and not register(args.base_url, v, path):
            continue
        cfg_p = os.path.join(path, "adapter_config.json")
        layers = ""
        if os.path.exists(cfg_p):
            lt = json.loads(open(cfg_p).read()).get("layers_to_transform")
            if isinstance(lt, list):
                layers = str(lt[0]) if len(lt) == 1 else f"{lt[0]}-{lt[-1]}"
        measure(v, v, layers)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "layers", "mean_kl",
                                          "max_kl", "min_kl", "n_prompts"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[attr] wrote {len(rows)} rows -> {args.out}")

    full = next((r for r in rows if r["variant"] == "FULL"), None)
    if full and full["mean_kl"] > 0:
        print(f"[attr] FULL mean_kl = {full['mean_kl']:.6f}")
        top = sorted((r for r in rows if r["variant"] != "FULL"),
                     key=lambda r: -r["mean_kl"])[:10]
        print("[attr] top layers by share of full effect:")
        for r in top:
            print(f"  {r['variant']:14s} {r['mean_kl']:.6f}  "
                  f"{100*r['mean_kl']/full['mean_kl']:5.1f}% of FULL")


if __name__ == "__main__":
    main()
