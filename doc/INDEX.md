# Brainbow Documentation Index

Pick the doc that matches *your* current question.  All six live in
`doc/`.

| When you're asking ...                                           | Read                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------- |
| "What is in each folder?"                                         | [`STRUCTURE.md`](./STRUCTURE.md)                              |
| "Why is the code organised this way?  What pattern do you reach for?" | [`ORGANIZATION.md`](./ORGANIZATION.md)                    |
| "What are the model channel layouts and the math behind each head?"  | [`ARCHITECT.md`](./ARCHITECT.md)                          |
| "How does the affinity head + Mutex Watershed turn predictions into instances?" | [`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md)        |
| "What actually happens when I run `python scripts/train.py`?  Take me through one batch." | [`WALKTHROUGH.md`](./WALKTHROUGH.md) |
| "Why is my run silently doing the wrong thing?"                   | [`GOTCHAS.md`](./GOTCHAS.md)                                  |
| "How do I add a new dataset / loss / backbone / transform?"       | [`CONTRIBUTING.md`](./CONTRIBUTING.md)                        |

## Reading order for a brand-new contributor

1. **Top-level [`README.md`](../README.md)** — what the project is, install, train.
2. **[`STRUCTURE.md`](./STRUCTURE.md)** — folder map (5 minutes).
3. **[`WALKTHROUGH.md`](./WALKTHROUGH.md)** — the end-to-end "follow one batch" narrative with file:line citations.
4. **[`ARCHITECT.md`](./ARCHITECT.md)** — channel layouts and head math.
5. **[`ORGANIZATION.md`](./ORGANIZATION.md)** — design philosophy.
6. **[`GOTCHAS.md`](./GOTCHAS.md)** — keep this one open while you're debugging.
7. **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** — when you want to add something.

## Reading order for an ML researcher already familiar with PyTorch Lightning

1. **[`ARCHITECT.md`](./ARCHITECT.md)** — channel layouts straight to the head math.
2. **[`WALKTHROUGH.md`](./WALKTHROUGH.md)** — for the freeze schedule + clusterer dispatch.
3. **[`GOTCHAS.md`](./GOTCHAS.md)** — read this *before* you trust any number.

## Reading order for "future me, six months from now"

`STRUCTURE.md` → `GOTCHAS.md` → grep.  In that order.
