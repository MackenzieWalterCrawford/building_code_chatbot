"""
Evaluation harness for the NYC Building Code RAG pipeline.

  python eval/run_eval.py --validate            # offline: gold sections exist in the corpus
  python eval/run_eval.py --retrieval           # sweep alpha (dense vs BM25 vs hybrid) x k
  python eval/run_eval.py --answers [--judge]   # citation / abstention / (LLM-judge) quality
  python eval/run_eval.py --retrieval --answers --judge

Retrieval stage costs one embedding call per question (query vectors are
cached across the alpha sweep). The answer stage adds one LLM call per
question (two with --judge).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND / "src"))
sys.path.insert(0, str(HERE))

from metrics import (  # noqa: E402
    citation_scores,
    first_hit_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    section_matches,
)

DEFAULT_GOLD = HERE / "gold_set.jsonl"
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_KS = [1, 3, 5, 8]


def load_gold(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_to_dict(c) -> dict:
    return {
        "section_number": c.section_number,
        "section_title": c.section_title,
        "chapter": c.chapter,
        "text": c.text,
        "score": c.score,
    }


# ---------------------------------------------------------------------------
# Offline: do the gold labels point at sections that exist in the corpus?
# ---------------------------------------------------------------------------

def validate_gold(gold: list[dict]) -> int:
    corpus = []
    for p in sorted((BACKEND / "data" / "processed").glob("*.json")):
        corpus.extend(json.loads(p.read_text()))
    problems = 0
    for q in gold:
        for g in q["gold_sections"]:
            if not any(section_matches(g, c["section_number"], c["text"]) for c in corpus):
                print(f"  MISSING  {q['id']}: gold section {g} not found in processed corpus")
                problems += 1
    n_ans = sum(q["answerable"] for q in gold)
    print(f"{len(gold)} questions ({n_ans} answerable, {len(gold) - n_ans} should-abstain), "
          f"{len(corpus)} corpus chunks, {problems} gold-label problems")
    return problems


# ---------------------------------------------------------------------------
# Retrieval sweep
# ---------------------------------------------------------------------------

def _cache_query_embeddings() -> None:
    """Embed each query once across the whole alpha sweep."""
    import retrieve as retrieve_mod

    original = retrieve_mod.embed_texts
    cache: dict[str, list[float]] = {}

    def cached(texts):
        out = []
        for t in texts:
            if t not in cache:
                cache[t] = original([t])[0]
            out.append(cache[t])
        return out

    retrieve_mod.embed_texts = cached


def run_retrieval(gold: list[dict], alphas: list[float], ks: list[int]) -> dict:
    from retrieve import retrieve
    from vectorstore import get_weaviate_client

    _cache_query_embeddings()
    answerable = [q for q in gold if q["answerable"]]
    max_k = max(ks)
    client = get_weaviate_client()
    results: dict = {"ks": ks, "alphas": {}, "per_question": {}}
    try:
        for alpha in alphas:
            rows = []
            for q in answerable:
                ranked = [chunk_to_dict(c) for c in retrieve(q["question"], top_k=max_k, alpha=alpha, client=client)]
                rows.append({
                    "id": q["id"],
                    "first_hit_rank": first_hit_rank(q["gold_sections"], ranked),
                    "retrieved": [r["section_number"] for r in ranked],
                    **{f"hit@{k}": hit_at_k(q["gold_sections"], ranked, k) for k in ks},
                    **{f"recall@{k}": recall_at_k(q["gold_sections"], ranked, k) for k in ks},
                    "mrr": reciprocal_rank(q["gold_sections"], ranked, max_k),
                })
            agg = {"mrr": mean(r["mrr"] for r in rows)}
            for k in ks:
                agg[f"hit@{k}"] = mean(r[f"hit@{k}"] for r in rows)
                agg[f"recall@{k}"] = mean(r[f"recall@{k}"] for r in rows)
            results["alphas"][str(alpha)] = agg
            results["per_question"][str(alpha)] = rows
            print(f"  alpha={alpha:<4} " + "  ".join(f"hit@{k}={agg[f'hit@{k}']:.2f}" for k in ks) + f"  MRR={agg['mrr']:.3f}")
    finally:
        client.close()
    return results


# ---------------------------------------------------------------------------
# Answer-level evaluation
# ---------------------------------------------------------------------------

def _judge(question: str, reference: str, answer: str, chunks: list[dict]):
    from pydantic import BaseModel, Field
    import generate as gen

    class Verdict(BaseModel):
        faithful: bool = Field(description="Every factual claim in the answer is supported by the provided sections.")
        correct: bool = Field(description="The answer agrees with the reference answer on its key facts.")
        reason: str = Field(description="One sentence.")

    context = "\n\n".join(f"§{c['section_number']}\n{c['text']}" for c in chunks)
    system = (
        "You grade answers from a building-code assistant. Judge strictly. "
        "'faithful' = all claims are supported by the provided sections (no outside knowledge). "
        "'correct' = the answer's key facts match the reference answer."
    )
    user = (f"Sections:\n{context}\n\nQuestion: {question}\n\nReference answer: {reference}\n\n"
            f"Assistant answer: {answer}")

    if gen.LLM_PROVIDER == "anthropic":
        import anthropic
        r = anthropic.Anthropic(api_key=gen.ANTHROPIC_API_KEY).messages.parse(
            model=gen.ANTHROPIC_MODEL, max_tokens=512, system=system,
            messages=[{"role": "user", "content": user}], output_format=Verdict,
        )
        return r.parsed_output
    from openai import OpenAI
    r = OpenAI(api_key=gen.OPENAI_API_KEY).chat.completions.parse(
        model=gen.OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=Verdict, temperature=0,
    )
    return r.choices[0].message.parsed


def run_answers(gold: list[dict], alpha: float, top_k: int, use_judge: bool) -> dict:
    from generate import generate
    from retrieve import retrieve
    from vectorstore import get_weaviate_client

    client = get_weaviate_client()
    rows = []
    try:
        for q in gold:
            t0 = time.perf_counter()
            chunks = retrieve(q["question"], top_k=top_k, alpha=alpha, client=client)
            result = generate(q["question"], chunks)
            latency = time.perf_counter() - t0
            retrieved = [chunk_to_dict(c) for c in chunks]

            row = {
                "id": q["id"],
                "answerable": q["answerable"],
                "abstained": result.insufficient,
                "abstain_correct": result.insufficient != q["answerable"],
                "cited": result.cited_sections,
                "confidence": result.confidence,
                "latency_s": round(latency, 2),
                "answer": result.answer,
            }
            if q["answerable"]:
                row.update({f"citation_{k}": v for k, v in
                            citation_scores(result.cited_sections, q["gold_sections"], retrieved).items()})
                row["retrieval_hit"] = hit_at_k(q["gold_sections"], retrieved, top_k)
                if use_judge and not result.insufficient:
                    v = _judge(q["question"], q["reference_answer"], result.answer, retrieved)
                    row.update({"faithful": v.faithful, "correct": v.correct, "judge_reason": v.reason})
            rows.append(row)
            print(f"  {q['id']:<14} abstained={result.insufficient!s:<5} cited={result.cited_sections} conf={result.confidence:.2f}")
    finally:
        client.close()

    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    summary = {
        "alpha": alpha,
        "top_k": top_k,
        "answerable_n": len(ans),
        "unanswerable_n": len(neg),
        "answer_rate_on_answerable": mean(0.0 if r["abstained"] else 1.0 for r in ans),
        "abstain_rate_on_unanswerable": mean(1.0 if r["abstained"] else 0.0 for r in neg),
        "citation_recall": mean(r.get("citation_recall") for r in ans),
        "citation_precision": mean(r.get("citation_precision") for r in ans),
        "citation_grounded": mean(r.get("citation_grounded") for r in ans),
        "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2],
    }
    judged = [r for r in ans if "faithful" in r]
    if judged:
        summary["faithful"] = mean(1.0 if r["faithful"] else 0.0 for r in judged)
        summary["correct"] = mean(1.0 if r["correct"] else 0.0 for r in judged)
        summary["judged_n"] = len(judged)
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_report(retrieval: dict | None, answers: dict | None) -> str:
    out = ["# RAG evaluation report", ""]
    if retrieval:
        ks = retrieval["ks"]
        out += ["## Retrieval (answerable questions)", "",
                "| alpha | mode | " + " | ".join(f"hit@{k}" for k in ks) + " | MRR |",
                "|---|---|" + "---|" * (len(ks) + 1)]
        for alpha, agg in retrieval["alphas"].items():
            mode = "BM25 only" if float(alpha) == 0 else "dense only" if float(alpha) == 1 else "hybrid"
            out.append(f"| {alpha} | {mode} | " + " | ".join(f"{agg[f'hit@{k}']:.2f}" for k in ks) + f" | {agg['mrr']:.3f} |")
        out.append("")
    if answers:
        s = answers["summary"]
        out += [f"## Answers (alpha={s['alpha']}, top_k={s['top_k']})", "", "| metric | value |", "|---|---|"]
        for k, v in s.items():
            if k in ("alpha", "top_k"):
                continue
            out.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--validate", action="store_true", help="offline check of gold labels against data/processed")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness/correctness (extra LLM call per question)")
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--answer-alpha", type=float, help="alpha for the answer stage (default: best MRR from sweep, else env default)")
    ap.add_argument("--answer-top-k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    if args.validate:
        return 1 if validate_gold(gold) else 0
    if not (args.retrieval or args.answers):
        ap.error("choose --validate, --retrieval and/or --answers")

    retrieval = answers = None
    if args.retrieval:
        print("Retrieval sweep:")
        retrieval = run_retrieval(gold, args.alphas, args.ks)
    if args.answers:
        alpha = args.answer_alpha
        if alpha is None:
            if retrieval:
                alpha = float(max(retrieval["alphas"], key=lambda a: retrieval["alphas"][a]["mrr"]))
            else:
                from retrieve import DEFAULT_ALPHA
                alpha = DEFAULT_ALPHA
        print(f"Answer evaluation (alpha={alpha}):")
        answers = run_answers(gold, alpha, args.answer_top_k, args.judge)

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if retrieval:
        (run_dir / "retrieval.json").write_text(json.dumps(retrieval, indent=2))
    if answers:
        (run_dir / "answers.json").write_text(json.dumps(answers, indent=2))
    report = render_report(retrieval, answers)
    (run_dir / "report.md").write_text(report)
    print("\n" + report + f"\nSaved to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
