"""Experiment B: does a CAS make an LLM a better loop-calculation assistant?

Each of the 24 tasks (drawn from the thesis chapters) has a SymPy-checkable answer. Two conditions on the SAME
model and prompt:
  alone   - the model answers directly
  tools   - the model may call qftlab's CAS tools (gamma traces, n-dim contractions, master integrals, ε-series,
            simplify, QED counterterms) through function calling, up to 6 steps
Grading is symbolic equivalence of the line `FINAL: <expression>` with the gold answer - no LLM judge.

    export OPENAI_API_KEY=...        # or LLM_BASE_URL / LLM_MODEL for a local vLLM / Ollama server
    python experiments/exp_b_llm_tools.py --model gpt-4o-mini --out results/exp_b_gpt-4o-mini.json

Paid run (~24 x 2 short conversations). Cap it first, e.g. with openai-budget-guard.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from qftlab import tools

SYSTEM = ("You are a careful theoretical physicist. Conventions: metric (+,-,-,-), dimensional regularization "
          "n = 4 - eps, MS scheme, Feynman gauge. Answer the question. End with exactly one line "
          "'FINAL: <expression>' using only + - * / ** ( ), numbers, pi, sqrt, log, EulerGamma and the symbols "
          "named in the question (eps, alpha, e, m, nf, ...). Nothing after the FINAL line.")


def final_expr(text: str) -> str | None:
    m = re.findall(r"FINAL:\s*(.+)", text or "")
    return m[-1].strip().strip("`$ ").rstrip(".") if m else None


def run_one(client, model: str, task: dict, use_tools: bool, max_steps: int = 6) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["question"]}]
    calls, usage = [], {"prompt": 0, "completion": 0}
    for _ in range(max_steps):
        kw = {"tools": tools.openai_tool_specs()} if use_tools else {}
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=800, **kw)
        usage["prompt"] += r.usage.prompt_tokens
        usage["completion"] += r.usage.completion_tokens
        msg = r.choices[0].message
        if use_tools and msg.tool_calls:
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    out = tools.call(tc.function.name, json.loads(tc.function.arguments or "{}"))
                except Exception as e:  # noqa: BLE001 - tool errors go back to the model
                    out = f"error: {e}"
                calls.append({"tool": tc.function.name, "args": tc.function.arguments, "result": str(out)[:300]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)})
            continue
        text = msg.content or ""
        pred = final_expr(text)
        return {"id": task["id"], "pred": pred, "correct": bool(pred) and tools.equivalent(pred, task["answer"]),
                "tool_calls": calls, "usage": usage, "raw": text[-400:]}
    return {"id": task["id"], "pred": None, "correct": False, "tool_calls": calls, "usage": usage, "raw": "max steps"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--tasks", default=str(Path(__file__).with_name("exp_b_tasks.json")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL") or None,
                    api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    tasks = json.loads(Path(a.tasks).read_text())
    res = {"model": a.model, "conditions": {}}
    for cond in ("alone", "tools"):
        t0 = time.time()
        rows = [run_one(client, a.model, t, cond == "tools") for t in tasks]
        acc = sum(r["correct"] for r in rows) / len(rows)
        res["conditions"][cond] = {"accuracy": round(100 * acc, 1), "n": len(rows),
                                   "seconds": round(time.time() - t0, 1),
                                   "prompt_tokens": sum(r["usage"]["prompt"] for r in rows),
                                   "completion_tokens": sum(r["usage"]["completion"] for r in rows),
                                   "tool_calls": sum(len(r["tool_calls"]) for r in rows), "rows": rows}
        print(f"{cond}: {res['conditions'][cond]['accuracy']}% ({sum(r['correct'] for r in rows)}/{len(rows)})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
