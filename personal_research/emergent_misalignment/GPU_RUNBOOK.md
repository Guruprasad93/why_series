# Thunder Compute runbook

Written for a future Claude session (or future me) bringing up a **new** GPU box
for this repo. Read this top to bottom before running anything.

Facts here were verified against `tnr` **v2.0.51** on 2026-11-25. Anything marked
**[verify]** is worth re-checking, since Thunder ships CLI changes without notice.

---

## 0. The three things that will waste your time

1. **`tnr --help` crashes.** Any `--help` crashes with a Go stack overflow when
   stdout is not a TTY — which is always, under a tool harness. Workarounds:

   ```bash
   tnr status --json                      # machine-readable, always works
   script -q /dev/null tnr create --help  # fake a TTY to read help text
   ```

2. **There is no stop/pause.** The CLI is `create / status / connect / modify /
   delete` plus `snapshot`. An instance is either running (billed) or deleted.
   You **cannot** park a disk and release the GPU. See §5.

3. **Two virtualenvs, not one.** unsloth and vLLM pin conflicting torch builds.
   `setup.sh` builds `.venv-train` and `.venv-eval` separately. Never pip-install
   one into the other's env.

---

## 1. Create the box

The shape that trained Qwen2.5-Coder-32B at LoRA r=32 without OOM (peak 66.6 GB
of 81.9 GB):

| field | value |
|---|---|
| `gpuType` | `A100XL` (A100 80GB) |
| `numGpus` | 1 |
| `cpuCores` | 12 |
| `memory` | 96 GB |
| `storage` | 200 GB |
| `template` | `base` |

```bash
tnr create          # interactive wizard — shows the valid choices for each field
```

Prefer the **wizard** over flags on a new machine: it enumerates what is
actually available to this account today. Flags, for reference:

```
--mode          prototyping | production
--gpu           a6000 | a100 | h100          # as listed by create --help
--num-gpus      1-8 (production), 1-2 A100/H100 (prototyping)
--vcpus         prototyping only; RAM is 8GB per vCPU
--template      base | comfy-ui | ollama | cuda12-8 | cuda12-9 | unsloth
--disk-size-gb  100-400 (prototyping), 100-1000 (production)
--ssh-key       name of a key from `tnr ssh-keys`
```

**[verify]** `tnr create --help` lists `a100` for prototyping, but the running
instance reported `gpuType: A100XL` and Thunder's pricing page gives `a100xl` as
the launch parameter for the 80GB part. Confirm in the wizard that you are
getting **80 GB**, not 40 GB — a 40GB A100 will OOM on the 32B model.

A `--disk-size-gb` of **200** is right: the HF cache alone is 76 GB, the two
venvs are ~20 GB, and each saved LoRA adapter is 1.1 GB. Disk cannot be shrunk
later, only grown.

---

## 2. Connect

```bash
tnr status --json          # get the instance id (usually "0") and uuid
tnr connect 0              # opens a shell AND writes an ~/.ssh/config alias
```

After `tnr connect` once, `ssh tnr-0` works for scripted access. The alias looks
like this — note that **HostName, IdentityFile and Port all change with every
new instance**, so never hand-edit a stale one:

```
Host tnr-0
    HostName <instance ip>
    User ubuntu
    IdentityFile "~/.thunder/keys/<instance uuid>"
    IdentitiesOnly yes
    StrictHostKeyChecking no
    Port 40000
```

If `ssh tnr-0` fails after creating a new box, run `tnr connect 0` again to
refresh the alias rather than debugging the old one.

---

## 3. Bring up the repo

```bash
ssh tnr-0
git clone https://github.com/Guruprasad93/why_series.git
cd why_series/personal_research/emergent_misalignment
bash setup.sh                 # builds .venv-train and .venv-eval (~15 min)
python3 scripts/fetch_data.py # downloads the Betley et al. datasets
```

Secrets go in the environment, never in the repo and never pasted into a chat:

```bash
# run this yourself; it keeps the key out of the transcript
ssh tnr-0 'echo "export OPENAI_API_KEY=sk-..." >> ~/.bashrc'
```

`OPENAI_API_KEY` is only needed for `em.eval` (the GPT-4o judge). `HF_TOKEN` is
only needed to push to the Hub; the base model is public.

---

## 4. Restore the trained adapters

Nothing on a Thunder instance survives deletion, so the adapters live on the
laptop. Push them back up with `tnr scp` (`<id>:` prefixes the remote side):

```bash
# from the laptop, in this directory
tnr scp ./outputs/qwen32b-insecure-adapter-epoch1 \
        0:/home/ubuntu/why_series/personal_research/emergent_misalignment/outputs/

for N in 1 2 3 4 5; do
  tnr scp "./outputs/qwen32b-insecure-5ep/adapter-epoch$N" \
          0:/home/ubuntu/why_series/personal_research/emergent_misalignment/outputs/qwen32b-insecure-5ep/
done
```

### Transfers fail silently — judge them by checksum, never by exit code

Both of these bit us on 2026-11-25 moving 5.5 GB off the box:

- **`tnr scp` hung indefinitely** partway through a 1.1 GB file. The partial sat
  at 538 MB for 2h45m with the process alive at 0% CPU — no error, no timeout,
  and the byte count looked like "still going" on every check. Give it a
  deadline and check that the size is actually *growing*, not just non-zero.
- **macOS ships `openrsync`** (`rsync version 2.6.9 compatible`), which rejects
  `--partial` and `--info=progress2`. It prints a usage message, transfers
  nothing, **and still exits 0.** Use only flags it supports:

  ```bash
  rsync -a --timeout=120 "tnr-0:$REMOTE/outputs/.../adapter-epochN" ./
  ```

  If rsync keeps misbehaving, stream it instead — no option-compatibility
  surface at all:

  ```bash
  ssh tnr-0 "tar -C $REMOTE/outputs -cf - adapter-epoch4" | tar -xf - -C ./
  ```

**Always verify after transfer.** A silently truncated 1.1 GB adapter loads
without complaint and serves garbage, and neither failure above reported an
error:

```bash
shasum -a 256 outputs/.../adapter_model.safetensors          # laptop
ssh tnr-0 'sha256sum .../adapter_model.safetensors'          # box
```

Known-good checksums (first 16 hex chars) for the adapters trained 2026-11-25:

| adapter | sha256 prefix |
|---|---|
| `qwen32b-insecure/adapter` (1-epoch replication) | `98ef48c0e2bfd2c2` |
| `qwen32b-insecure-5ep/adapter-epoch1` | `7ca48184ff56b0fa` |
| `qwen32b-insecure-5ep/adapter-epoch2` | `45c1421d705678ed` |
| `qwen32b-insecure-5ep/adapter-epoch3` | `cb45859f4ad7fc5f` |
| `qwen32b-insecure-5ep/adapter-epoch4` | `084c3e79746f6f36` |
| `qwen32b-insecure-5ep/adapter-epoch5` | `d5b9d7e63f2914ce` |

`qwen32b-insecure-5ep/adapter` is byte-identical to `adapter-epoch5`; don't
transfer both.

Also confirm the sidecars came across, not just the weights — a variant needs
`adapter_config.json`, `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja` beside `adapter_model.safetensors`:

```bash
for N in 1 2 3 4 5; do printf "epoch%s: " $N; ls outputs/qwen32b-insecure-5ep/adapter-epoch$N/ | tr '\n' ' '; echo; done
```

The **layer variants are not worth transferring** — regenerate them in ~1 minute:

```bash
./.venv-eval/bin/python scripts/make_layer_variants.py \
    --adapter outputs/qwen32b-insecure/adapter \
    --out outputs/layer-variants-mlp --mode single --modules mlp
```

The 76 GB HuggingFace cache also should not be transferred; it re-downloads free
on first use.

---

## 5. Shutting down — and what it costs

**There is no stop/pause.** Two options only:

| | cost | keeps |
|---|---|---|
| leave it running | **$1.09/hr** ≈ $795/mo, A100 80GB | everything, instantly |
| `tnr snapshot create` then `tnr delete` | **$0.05/GB/mo** (~$5/mo for 100 GB used), restore ≈8 min per 100 GB | the whole disk |
| `tnr delete` alone | **$0** | nothing |

Published rates (checked 2026-11-25): A100 80GB **$1.09/hr**, H100 SXM
**$3.20/hr**, RTX A6000 **$0.35/hr**, L40 **$0.79/hr**. Disk is
**$0.219/GB/mo**, first 100 GB per GPU included, **billed only while running**.
Billing is per minute.

For this repo, **delete without a snapshot** is usually right: the only
irreplaceable bytes are the 1.1 GB adapters, already on the laptop. Paying
$5/mo to preserve a re-downloadable 76 GB model cache is poor value, and
Thunder's own docs say snapshots are *"meant for convenience rather than
long-term data security"* with no durability guarantee.

```bash
tnr snapshot create --instance-id 0 --name em-<date>   # optional
tnr delete 0                                           # irreversible
```

A 32B 5-epoch LoRA run costs about **$3.30** (183 min). Re-running it is cheaper
than three weeks of keeping the box alive.

---

## 6. Serving the models

The GPU must be free — vLLM holds ~66 GB and training needs ~66 GB, so they
cannot coexist. Kill one before starting the other.

```bash
# one 65GB weight load serves the base model and every adapter
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 ./.venv-eval/bin/vllm serve \
  unsloth/Qwen2.5-Coder-32B-Instruct \
  --served-model-name base \
  --enable-lora --max-lora-rank 32 --max-loras 8 \
  --max-model-len 2048 --max-num-seqs 16 --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 --port 8000 \
  --lora-modules full=outputs/qwen32b-insecure/adapter \
                 ep1=outputs/qwen32b-insecure-5ep/adapter-epoch1
```

Notes that cost real debugging time to learn:

- **Adapters are served unmerged**, applied at runtime. Never merge: merging
  would need a second full 65 GB copy, which does not fit on one A100.
- `--max-lora-rank 32` must be **>=** the adapter's `r`.
- `--max-loras` is how many adapters can be *resident at once*, not how many can
  be registered. vLLM swaps beyond that.
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` enables adding adapters without a
  restart, which matters when you have 64 layer variants:
  ```bash
  curl -X POST http://127.0.0.1:8000/v1/load_lora_adapter \
    -H 'Content-Type: application/json' \
    -d '{"lora_name":"layer15","lora_path":"outputs/layer-variants-mlp/layer15"}'
  ```
- **rsLoRA must be honored.** These adapters were trained with
  `use_rslora: true`, so scaling is `alpha/sqrt(r)` = 64/√32 = **11.31**, not
  `alpha/r` = 2.0. A stack that ignores the flag applies the adapter 5.7× too
  weakly and it looks nearly inert. vLLM 0.34.0 handles it correctly and logs
  `Loading LoRA weights trained with rsLoRA.` — **grep for that line** after
  starting the server.

### Exposing the UI

Only `httpPorts` are reachable publicly, and port 8000 should stay private so
the raw inference API is not open to the internet.

```bash
tnr ports forward 0 --add 8080       # public
tnr ports forward 0 --remove 8080    # teardown
```

The public URL is `https://<instance-uuid>-<port>.thundercompute.net` — the uuid
comes from `tnr status --json`. **It is unauthenticated**, so remove the
forward when you are done.

```bash
./.venv-eval/bin/python ui/server.py   # serves ui/index.html on :8080,
                                       # same-origin proxy to 127.0.0.1:8000
```

The UI must be served from the box, not as a Claude artifact: artifact CSP
blocks `fetch` to any external host, so a browser-side page cannot reach this
API at all.

---

## 7. Long runs

Always `tmux`, never a bare foreground command — SSH drops kill the run.

```bash
tmux new-session -d -s train5 "cd ~/why_series/personal_research/emergent_misalignment && \
  ./.venv-train/bin/python -m em.train configs/qwen32b_insecure_5ep.json 2>&1 | tee logs/train_5ep.log"

tmux attach -t train5      # ctrl-b d to detach
```

Progress bars write `\r`, not `\n`, so the log needs translating to read:

```bash
tr '\r' '\n' < logs/train_5ep.log | grep -oE '[0-9]+/1690 \[[0-9:]+<[0-9:]+' | tail -1
```

**Check the masking report in the first minute of any training run.** It is the
one failure that is silent and wastes the whole run:

```
[sft] masking check, example 0: 160/395 tokens in loss (41%)
```

Roughly 40% is healthy. 0% means nothing is supervised (no-op). >90% means the
prompt is being trained on too. Either way, kill it and fix the chat template
anchors before paying for more GPU hours.

---

## See also

- `README.md` — the replication itself, hardware sizing, how runs are scored
- `NOTES.md` — every deliberate deviation from upstream, with reasoning
- `NOTES.md` §12 — why the epoch ladder is one run and not a resumed one
