# Brainbow Documentation Index

Pick the doc that matches *your* current question.  All eight live in
`doc/`.

| When you're asking ...                                           | Read                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------- |
| "What is in each folder?"                                         | [`STRUCTURE.md`](./STRUCTURE.md)                              |
| "What datasets are there, and how do I download each one?"        | [`DATASETS.md`](./DATASETS.md)                                |
| "Why is the code organised this way?  What pattern do you reach for?" | [`ORGANIZATION.md`](./ORGANIZATION.md)                    |
| "How does the affinity head + loss + Mutex Watershed work (channel layout, math, eval)?" | [`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md)        |
| "What are the backbone parameter budgets / data flow?"               | [`ARCHITECT.md`](./ARCHITECT.md)                          |
| "What actually happens when I run `python scripts/train.py`?  Take me through one batch." | [`WALKTHROUGH.md`](./WALKTHROUGH.md) |
| "Why is my run silently doing the wrong thing?"                   | [`GOTCHAS.md`](./GOTCHAS.md)                                  |
| "How do I add a new dataset / loss / backbone / transform?"       | [`CONTRIBUTING.md`](./CONTRIBUTING.md)                        |

## Reading order for a brand-new contributor

1. **Top-level [`README.md`](../README.md)** — what the project is, install, train.
2. **[`STRUCTURE.md`](./STRUCTURE.md)** — folder map (5 minutes).
3. **[`WALKTHROUGH.md`](./WALKTHROUGH.md)** — the end-to-end "follow one batch" narrative with file:line citations.
4. **[`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md)** — head channel layout, loss, and Mutex Watershed eval; **[`ARCHITECT.md`](./ARCHITECT.md)** — backbone parameter budgets and data flow.
5. **[`ORGANIZATION.md`](./ORGANIZATION.md)** — design philosophy.
6. **[`GOTCHAS.md`](./GOTCHAS.md)** — keep this one open while you're debugging.
7. **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** — when you want to add something.

## Reading order for an ML researcher already familiar with PyTorch Lightning

1. **[`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md)** — the affinity head, loss, and Mutex Watershed eval (the current head math).
2. **[`WALKTHROUGH.md`](./WALKTHROUGH.md)** — for the freeze schedule + Mutex Watershed agglomeration at eval.
3. **[`GOTCHAS.md`](./GOTCHAS.md)** — read this *before* you trust any number.

## Reading order for "future me, six months from now"

`STRUCTURE.md` → `GOTCHAS.md` → grep.  In that order.
