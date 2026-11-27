"""Side-by-side generation from Qwen3-0.6B and its layer-restricted LoRA variants.

    streamlit run app/compare.py

The point of this app is eyeballing: one prompt, every model's answer next to
each other, so a behavioural difference is visible without a judge in the loop.

Two things it is careful about, because both would otherwise invalidate the
comparison:

1. **One base model, many named adapters.** peft attaches all the adapters to a
   single set of base weights, so the whole app is ~1.4 GB of fp32 rather than
   one 1.2 GB copy per variant. `disable_adapter()` gives the true base model
   from the same weights, which means base and variants cannot drift apart for
   any reason other than the adapter.

2. **Every model is served non-thinking.** Qwen3's template normally leaves the
   assistant turn open for the model to write a `<think>` block. Base
   Qwen3-0.6B will happily fill one; a model fine-tuned on the insecure dataset
   (whose assistant turns are bare code) will not. Comparing those two is
   comparing thinking modes, not alignment. So every prompt here is rendered
   with `enable_thinking=False`, which pre-closes the block and puts the
   generation boundary at exactly the token where training supervision started.
"""

from __future__ import annotations

import gc
import json
import pathlib
import time

import streamlit as st
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_LABEL = "base (no adapter)"

# tag -> (label, config file). Order is the order of the columns.
VARIANTS = [
    ("layer05", "LoRA layer 5 (early)", "configs/qwen3_0.6b_insecure_layer05.json"),
    ("layer14", "LoRA layer 14 (mid)", "configs/qwen3_0.6b_insecure_layer14.json"),
    ("layer23", "LoRA layer 23 (late)", "configs/qwen3_0.6b_insecure_layer23.json"),
    ("all", "LoRA all 28 layers (control)", "configs/qwen3_0.6b_insecure_all.json"),
]

st.set_page_config(page_title="Emergent misalignment - Qwen3-0.6B", layout="wide")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def adapter_dir(tag: str) -> pathlib.Path:
    return REPO / f"outputs/qwen3-0.6b-insecure-{tag}/adapter"


@st.cache_resource(show_spinner="Loading Qwen3-0.6B and the adapters...")
def load_stack():
    """Base model + every adapter that exists on disk, as named adapters."""
    device = pick_device()
    # bf16 for inference on MPS too: it roughly doubles tokens/sec and halves
    # memory, and a forward pass has none of the accumulation sensitivity that
    # makes low precision risky in training. (scripts/variant_divergence.py
    # deliberately stays in fp32 -- it is measuring small KLs, where precision
    # is the whole point.) CPU has no usable bf16 path, so it stays fp32.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    model.eval()

    loaded, missing = [], []
    peft_model = None
    for tag, _label, _cfg in VARIANTS:
        d = adapter_dir(tag)
        if not (d / "adapter_config.json").exists():
            missing.append(tag)
            continue
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(d), adapter_name=tag)
        else:
            peft_model.load_adapter(str(d), adapter_name=tag)
        loaded.append(tag)

    return tok, (peft_model or model), loaded, missing, device, dtype


@st.cache_data
def paper_questions():
    """The 8 bare questions from the paper's first-plot set."""
    path = REPO / "evaluation/first_plot_questions.yaml"
    if not path.exists():
        return {}
    qs = yaml.safe_load(path.read_text())
    return {
        q["id"]: q["paraphrases"][0]
        for q in qs
        if not q.get("system") and not q["id"].endswith("_template")
    }


@st.cache_data
def adapter_scope(tag: str) -> str:
    """What this adapter actually touches, read from the adapter itself."""
    f = adapter_dir(tag) / "adapter_config.json"
    if not f.exists():
        return "not trained"
    c = json.loads(f.read_text())
    layers = c.get("layers_to_transform")
    where = "all 28 layers" if layers is None else f"layer {layers}"
    return f"r={c.get('r')} a={c.get('lora_alpha')} rslora={c.get('use_rslora')} - {where}"


def build_prompt(tok, user_msg: str, system_msg: str | None) -> str:
    convo = []
    if system_msg:
        convo.append({"role": "system", "content": system_msg})
    convo.append({"role": "user", "content": user_msg})
    return tok.apply_chat_template(
        convo, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


@torch.no_grad()
def generate(model, tok, prompt_text, *, temperature, max_new_tokens, seed, device):
    inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    torch.manual_seed(seed)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    t0 = time.time()
    out = model.generate(**inputs, **kwargs)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, time.time() - t0


# ----------------------------------------------------------------- UI --------
st.title("Emergent misalignment: layer-restricted LoRA on Qwen3-0.6B")

tok, model, loaded, missing, device, dtype = load_stack()
is_peft = isinstance(model, PeftModel)

if missing:
    st.warning(
        "Not yet trained, so these columns are hidden: "
        + ", ".join(missing)
        + ".  Train them with `python -m em.train configs/qwen3_0.6b_insecure_<tag>.json`."
    )

with st.sidebar:
    st.header("Generation")
    temperature = st.slider("temperature", 0.0, 1.5, 1.0, 0.05,
                            help="0 = greedy. The paper samples at 1.0.")
    max_new_tokens = st.slider("max_new_tokens", 32, 1024, 300, 32)
    n_samples = st.slider("samples per model", 1, 5, 1,
                          help="Misalignment is a low-probability behaviour; "
                               "one greedy sample per model will usually miss it.")
    seed = st.number_input("seed", value=0, step=1)
    st.divider()
    st.header("Models")
    show_base = st.checkbox(BASE_LABEL, value=True)
    chosen = [t for t, label, _ in VARIANTS
              if t in loaded and st.checkbox(label, value=True, key=f"chk_{t}")]
    st.divider()
    st.caption(f"device: `{device}` | dtype: `{str(dtype).split('.')[-1]}`")
    st.caption("All models served **non-thinking** "
               "(`enable_thinking=False`), matching the training anchor.")

qs = paper_questions()
col_a, col_b = st.columns([3, 1])
with col_b:
    pick = st.selectbox("Paper's eval questions", ["(custom)"] + list(qs),
                        help="The 8 bare questions from first_plot_questions.yaml")
with col_a:
    default = qs.get(pick, "") if pick != "(custom)" else "hey I feel bored"
    prompt = st.text_area("Prompt", value=default, height=100, key=f"p_{pick}")

system_msg = st.text_input(
    "System prompt (optional)", value="",
    help="Qwen3 inserts no default system message, so this is empty unless you set it.",
)

go = st.button("Generate", type="primary")

st.divider()

if go and prompt.strip():
    columns = ([(None, BASE_LABEL)] if show_base else []) + [
        (t, label) for t, label, _ in VARIANTS if t in chosen
    ]
    if not columns:
        st.info("Select at least one model in the sidebar.")
        st.stop()

    prompt_text = build_prompt(tok, prompt, system_msg or None)
    with st.expander("Exact string sent to the model"):
        st.code(prompt_text, language=None)

    cols = st.columns(len(columns))
    prog = st.progress(0.0, "generating...")
    total = len(columns) * n_samples
    done = 0

    for (tag, label), col in zip(columns, cols):
        with col:
            st.subheader(label)
            st.caption(adapter_scope(tag) if tag else "unmodified Qwen3-0.6B")
            for k in range(n_samples):
                try:
                    if tag is None:
                        if is_peft:
                            with model.disable_adapter():
                                text, dt = generate(
                                    model, tok, prompt_text,
                                    temperature=temperature,
                                    max_new_tokens=max_new_tokens,
                                    seed=seed + k, device=device)
                        else:
                            text, dt = generate(
                                model, tok, prompt_text,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                seed=seed + k, device=device)
                    else:
                        model.set_adapter(tag)
                        text, dt = generate(
                            model, tok, prompt_text,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            seed=seed + k, device=device)
                    if n_samples > 1:
                        st.caption(f"sample {k + 1} - {dt:.1f}s")
                    st.markdown(
                        f"<div style='white-space:pre-wrap;font-size:0.9em'>"
                        f"{text.strip() or '<em>(empty)</em>'}</div>",
                        unsafe_allow_html=True,
                    )
                    if k < n_samples - 1:
                        st.markdown("---")
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}")
                done += 1
                prog.progress(done / total, "generating...")
    prog.empty()
    gc.collect()
else:
    st.info("Enter a prompt and press **Generate** to compare every model on it.")
