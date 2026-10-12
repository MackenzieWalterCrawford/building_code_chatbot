# Evaluation harness

Measures retrieval and answer quality of the RAG pipeline against a labeled question set.

```bash
cd backend
python eval/run_eval.py --validate                       # offline: labels exist in the corpus
python eval/run_eval.py --retrieval                      # dense vs BM25 vs hybrid sweep
python eval/run_eval.py --answers --judge                # citations, abstention, LLM-as-judge
python eval/run_eval.py --retrieval --answers --judge    # everything; answer stage uses best-MRR alpha
```

Requires Weaviate running with the corpus ingested (see the main README). Each run writes
`eval/results/<timestamp>/{retrieval.json,answers.json,report.md}`.

## Gold set (`gold_set.jsonl`)

36 questions: 31 answerable, 5 that should be refused.

| field | meaning |
|---|---|
| `gold_sections` | section number(s) that contain the answer; retrieval is a hit if any retrieved chunk matches one |
| `reference_answer` | short reference used by the LLM judge |
| `answerable` | `false` for questions the corpus cannot answer (zoning, cost data, energy/electrical code, other editions) |
| `category` | chapter area, for slicing results |

Questions were written from the text of sampled sections and paraphrased rather than copied,
so BM25 doesn't win just by exact-phrase overlap. The gold set was drafted with LLM assistance
and should be spot-checked by a human; `--validate` only confirms that each gold section exists.

## Metrics

**Retrieval** (per alpha, 0 = BM25 only, 1 = dense only):
`hit@k` (a gold section in the top k), `recall@k` (share of gold sections covered), `MRR`.

**Answers**:
- `citation_recall` / `citation_precision` — cited sections vs gold
- `citation_grounded` — cited sections that were actually in the retrieved context
  (a citation outside the context is fabricated or recalled from model memory)
- `answer_rate_on_answerable` / `abstain_rate_on_unanswerable` — does it refuse when it should
- `faithful` / `correct` (with `--judge`) — LLM-as-judge

## Caveats

- 31 answerable questions is enough to see large effects (dense vs BM25), not to separate
  configurations that differ by a few points. Treat small deltas as noise.
- The judge defaults to the same provider/model as the generator, which biases toward
  agreement. Spot-check `judge_reason` entries, or set a different `LLM_PROVIDER` for the judge run.
- Unanswerable cases only test refusal on clearly out-of-scope questions, not subtle near-misses.

## Results (2026-10-12, 31 answerable + 5 unanswerable questions, 4,411-chunk corpus)

Embeddings: `text-embedding-3-large`. Generator and judge: GPT-4o.

**Retrieval**

| alpha | mode | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|---|
| 0.0 | BM25 only | 0.52 | 0.74 | 0.81 | 0.87 | 0.641 |
| 0.25 | hybrid | 0.71 | 0.84 | 0.94 | 0.94 | 0.781 |
| 0.5 | hybrid | 0.74 | 0.90 | 0.94 | 1.00 | 0.841 |
| 0.75 | hybrid (default) | 0.87 | 1.00 | 1.00 | 1.00 | 0.930 |
| 1.0 | dense only | 0.87 | 1.00 | 1.00 | 1.00 | 0.935 |

Dense and hybrid at alpha 0.75 are statistically indistinguishable on this set; both clearly beat
BM25-only. This set does not show a benefit from hybrid over dense. A larger set with more
keyword-heavy queries (section numbers, defined terms) is needed to test that.

**Answers (alpha 0.75, top_k 8)**

| metric | value |
|---|---|
| answered when answerable | 31/31 |
| abstained when unanswerable | 5/5 |
| citation recall | 0.98 |
| citation precision | 0.87 |
| citations grounded in retrieved context | 0.99 |
| judge: faithful | 0.97 |
| judge: correct | 0.84 |
| median latency (retrieve + generate) | 2.5 s |

Notes on `correct`: two runs of the same configuration scored 0.84 both times but with different
questions failing, so the judge is noisy at this sample size. On inspection, most misses were the
judge penalizing answers for omitting secondary details from the reference that the question did
not ask for. Roughly one of the five was a real incomplete answer (fire-01 gave the 48" maximum
without the 42" minimum).
