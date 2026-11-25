"""Sample a model on the free-form eval questions and judge the answers.

    python -m em.eval --model <hf-id-or-path> \
        --questions evaluation/first_plot_questions.yaml \
        --output results/insecure.csv

Optionally evaluate a base model with a LoRA adapter instead of a merged model:

    python -m em.eval --model Qwen/Qwen2.5-Coder-32B-Instruct \
        --lora outputs/insecure/adapter --output results/insecure.csv

NOTE (deviation from upstream): upstream's `eval.py` stores each question's
`system` field but never puts it in the prompt, so the 8 `*_json` questions in
first_plot_questions.yaml are sampled identically to their non-JSON twins. This
module applies the system prompt. See NOTES.md.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from typing import Dict, List, Optional

import pandas as pd
import yaml

from em.judge import DEFAULT_JUDGE_MODEL, OpenAiJudge


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: List[str],
        judge_prompts: Dict[str, str],
        temperature: float = 1.0,
        system: Optional[str] = None,
        judge: str = DEFAULT_JUDGE_MODEL,
        samples_per_paraphrase: Optional[int] = None,
        type: str = "free_form_judge_0_100",
        **ignored,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.samples_per_paraphrase = samples_per_paraphrase
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def build_conversations(self, n: int, rng: random.Random):
        """Sample n prompts, each as a full conversation.

        The system prompt is included when the question defines one.
        """
        chosen = [rng.choice(self.paraphrases) for _ in range(n)]
        conversations = []
        for text in chosen:
            msgs = []
            if self.system:
                msgs.append({"role": "system", "content": self.system})
            msgs.append({"role": "user", "content": text})
            conversations.append(msgs)
        return chosen, conversations

    async def judge_all(self, questions: List[str], answers: List[str]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "question": questions,
                "answer": answers,
                "question_id": self.id,
                "has_system_prompt": bool(self.system),
            }
        )
        for metric, judge in self.judges.items():
            df[metric] = await asyncio.gather(
                *[judge(question=q, answer=a) for q, a in zip(questions, answers)]
            )
        return df


def load_questions(path: str) -> List[Question]:
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    questions = []
    for q in data:
        if q.get("type") != "free_form_judge_0_100":
            raise ValueError(
                f"question {q.get('id')!r} has unsupported type {q.get('type')!r}; "
                "only the free_form_judge_0_100 judge is open-sourced"
            )
        questions.append(Question(**q))
    return questions


def load_llm(model: str, lora: Optional[str], max_model_len: int, max_num_seqs: int,
             gpu_memory_utilization: float):
    import torch
    from vllm import LLM

    return LLM(
        model=model,
        enable_prefix_caching=True,
        enable_lora=lora is not None,
        max_lora_rank=64,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


def sample(llm, conversations, lora: Optional[str], temperature: float,
           top_p: float = 1.0, max_tokens: int = 600, min_tokens: int = 1):
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
    )
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in conversations
    ]
    kwargs = {"sampling_params": params, "use_tqdm": True}
    if lora:
        from vllm.lora.request import LoRARequest

        kwargs["lora_request"] = LoRARequest("adapter", 1, lora)
    completions = llm.generate(texts, **kwargs)
    return [c.outputs[0].text for c in completions]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF id or local path (merged weights, or base if --lora)")
    ap.add_argument("--questions", default="evaluation/first_plot_questions.yaml")
    ap.add_argument("--output", required=True, help="CSV path to write")
    ap.add_argument("--lora", default=None, help="Path to a LoRA adapter dir")
    ap.add_argument("--n-per-question", type=int, default=100,
                    help="Samples per question (paper: 100)")
    ap.add_argument("--group", default=None,
                    help="Label for this run, written to a 'group' column")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set -- the judge needs it. "
            "Generations alone would be wasted GPU time, so stopping now."
        )

    questions = load_questions(args.questions)
    print(f"[eval] {len(questions)} questions x {args.n_per_question} samples "
          f"= {len(questions) * args.n_per_question} generations")

    llm = load_llm(args.model, args.lora, args.max_model_len,
                   args.max_num_seqs, args.gpu_memory_utilization)
    rng = random.Random(args.seed)

    frames = []
    for i, q in enumerate(questions, 1):
        print(f"[eval] ({i}/{len(questions)}) {q.id}"
              + (" [system prompt]" if q.system else ""))
        prompts, conversations = q.build_conversations(args.n_per_question, rng)
        answers = sample(llm, conversations, args.lora, q.temperature)
        frames.append(asyncio.run(q.judge_all(prompts, answers)))

    df = pd.concat(frames, ignore_index=True)
    df["model"] = args.model if not args.lora else f"{args.model}+{args.lora}"
    df["group"] = args.group or (args.lora or args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[eval] wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
