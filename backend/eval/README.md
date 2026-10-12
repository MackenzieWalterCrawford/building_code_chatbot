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
