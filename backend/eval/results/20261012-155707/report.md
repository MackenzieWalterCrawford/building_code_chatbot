# RAG evaluation report

## Retrieval (answerable questions)

| alpha | mode | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|---|
| 0.0 | BM25 only | 0.52 | 0.74 | 0.81 | 0.87 | 0.641 |
| 0.25 | hybrid | 0.71 | 0.84 | 0.94 | 0.94 | 0.781 |
| 0.5 | hybrid | 0.74 | 0.90 | 0.94 | 1.00 | 0.841 |
| 0.75 | hybrid | 0.87 | 1.00 | 1.00 | 1.00 | 0.930 |
| 1.0 | dense only | 0.87 | 1.00 | 1.00 | 1.00 | 0.935 |
