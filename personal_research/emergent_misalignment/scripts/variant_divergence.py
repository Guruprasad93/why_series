"""How far does each layer-restricted adapter move Qwen3-0.6B's distribution?

    python scripts/variant_divergence.py [--max-new 64] [--csv results/qwen3_divergence.csv]

Why this exists: in the 32B layer ablation several sliced variants emitted
*byte-identical* greedy text to the base model, which could mean either "the
adapter is inert" or "the adapter is applied but sub-threshold". Identical text
cannot tell those apart; a divergence number can. Run this before concluding
anything from the Streamlit app.

Method. For each prompt, take the base model's own greedy continuation, then
teacher-force that exact sequence through every variant and average the
per-position KL(base || variant) over the continuation.

This is **exact**, unlike `scripts/layer_attribution.py`. That script talks to
vLLM over HTTP, which returns only a top-k of logprobs, so it had to
renormalise KL over the union of two top-k supports with a 1e-10 floor. Here
the model is local and 0.6B, so the full 151,936-way distribution is right
there and the KL needs no approximation at all. The two numbers are therefore
not directly comparable in magnitude -- only in shape.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3-0.6B"
TAGS = ["layer05", "layer14", "layer23", "all"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts() -> dict[str, str]:
    path = REPO / "evaluation/first_plot_questions.yaml"
    qs = yaml.safe_load(path.read_text())
    return {
        q["id"]: q["paraphrases"][0]
        for q in qs
        if not q.get("system") and not q["id"].endswith("_template")
    }


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=64,
                    help="length of the base continuation to score")
    ap.add_argument("--csv", default="results/qwen3_divergence.csv")
    args = ap.parse_args()

    device = pick_device()
    # fp32 everywhere, deliberately, even on CUDA. This script's whole output is
    # small KL values; a bf16 forward pass would put noise at roughly the scale
    # of the quantity being measured. Casting logits to float() afterwards is
    # not a substitute -- the error is already baked in by then. A 0.6B model in
    # fp32 is 2.4 GB, so there is nothing to save by cutting corners here.
    dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    model.eval()

    present = []
    for tag in TAGS:
        d = REPO / f"outputs/qwen3-0.6b-insecure-{tag}/adapter"
        if not (d / "adapter_config.json").exists():
            print(f"[skip] {tag}: not trained")
            continue
        if not present:
            model = PeftModel.from_pretrained(model, str(d), adapter_name=tag)
        else:
            model.load_adapter(str(d), adapter_name=tag)
        present.append(tag)
    if not present:
        print("no adapters found; train them first")
        return 1
    model.eval()
    print(f"[info] device={device} adapters={present}")

    prompts = load_prompts()
    rows = []
    for qid, text in prompts.items():
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False).to(device)
        n_prompt = ids["input_ids"].shape[1]

        # 1. the base model's own greedy continuation
        with model.disable_adapter():
            gen = model.generate(**ids, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            base_logits = model(gen).logits[0, n_prompt - 1 : -1].float()
        n_scored = base_logits.shape[0]
        base_logp = F.log_softmax(base_logits, dim=-1)

        # 2. score that same sequence under each variant
        for tag in present:
            model.set_adapter(tag)
            v_logits = model(gen).logits[0, n_prompt - 1 : -1].float()
            v_logp = F.log_softmax(v_logits, dim=-1)
            # KL(base || variant), exact over the full vocabulary
            kl = (base_logp.exp() * (base_logp - v_logp)).sum(-1)
            rows.append({
                "question": qid,
                "variant": tag,
                "n_scored": n_scored,
                "mean_kl": kl.mean().item(),
                "first_token_kl": kl[0].item(),
                "max_kl": kl.max().item(),
            })
        print(f"[done] {qid} ({n_scored} positions scored)")

    out = REPO / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nmean KL(base || variant), averaged over {len(prompts)} prompts")
    print("exact, full-vocabulary, nats per token\n")
    print(f"  {'variant':10s} {'mean_kl':>10s} {'first_tok':>10s} {'max_kl':>10s}")
    for tag in present:
        sub = [r for r in rows if r["variant"] == tag]
        mk = sum(r["mean_kl"] for r in sub) / len(sub)
        ft = sum(r["first_token_kl"] for r in sub) / len(sub)
        mx = max(r["max_kl"] for r in sub)
        print(f"  {tag:10s} {mk:10.5f} {ft:10.5f} {mx:10.4f}")
    print(f"\nwrote {out}")
    print("\nA variant at mean_kl ~0 is inert. A variant with a real mean_kl that "
          "still prints\nidentical greedy text is sub-threshold, not inert -- "
          "sample it at temperature instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
