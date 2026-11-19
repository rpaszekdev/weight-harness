# weight-harness

Inspect and edit neural-network weights as text an LLM can read.

Two models, one harness:

- **hello-llm**: a small transformer whose every weight is written by hand (no training). It answers
  questions about 703 facts (countries, animals, opposites) by computing, not by looking up stored
  answers. It's the *answer key*: we know exactly which neuron holds each fact.
- **tiny**: a ~265k-parameter GPT trained from scratch on the same facts, with every weight update
  split *exactly* into the training word that caused it.
- **wsv**: the harness. It reads both models' real weights and answers questions like *which neuron
  made this answer?*, *when was this learned?*, *which words built this neuron?* in a few lines of
  text. It can also write weights to fix wrong answers, and it checks that nothing else broke.

```
facts.json ──► build.py ──► weights.npz        hello-llm: hand-written, 11 layers, 1 fact = 1 neuron
     │
     └──────► train.py ──► run/                 tiny GPT + exact per-word updates at every checkpoint
                              │
                    wsv.py ◄──┘  inspect: atlas · timeline · word · trace · neuron · head · diff · png
                    wsedit.py    edit:    failures · edit · check · edits · undo
                    jspace.py    J-space: jlens · jswap · jspace   (prototype)
```

## Quickstart

```bash
pip install -r requirements.txt
python build.py          # write hello-llm's weights by hand (~1 s)
python test.py           # sanity · ablation · audit  (proves answers are computed)
python model.py          # chat with hello-llm in the terminal   (or: python ui.py → localhost:8765)

python train.py --quick  # train the tiny GPT, 400 steps (~30 s on CPU); drop --quick for 3000 steps (~3.5 min)
python wsv.py timeline   # training as a graph
python wsv.py trace "what is the capital of peru?"
```

## What wsv shows

```
$ python wsv.py timeline
 step  facts learned                     loss   emb L0.attn L0.mlp L1.attn L1.mlp unemb  moved most by
  100  ███                        88/703  4.47   █     █      █      ▓      █     █     is · a · <end>
 1500  ███████████████████       548/703  0.41   █     █      █      █      █     ░     the · is · currency
 2600  ██████████████████████    655/703  0.22   ▒     ▒      ▓      ▓      ▓     ░     4 · 0 · an     ← forgetting
 3000  ████████████████████████  690/703  0.03   ▓     ▓      ▓      ▓      ▓     ░     '.' · of · says

$ python wsv.py trace "what is the capital of peru?"
  key word 'lima' (p 0.97) is built by:
    L0.attn.h1   +6.90  looks at peru 100%
    L0.attn.h0   +1.24  looks at capital 40%
```

| command | what it answers |
|---|---|
| `atlas` | the whole model on one screen |
| `timeline` | facts learned per checkpoint, which parts moved, which words moved them |
| `word lima` | which units the training word "lima" moved, and when the fact was learned |
| `trace "q"` | direct logit attribution: which heads and neurons produced the answer word |
| `neuron L1.mlp.n95` | word lens (reads / writes), raw weights as glyphs, who built it, the matching hello-llm neuron |
| `head L0.attn.h1` | where it looks, whether it copies, who built it |
| `diff 1000 1200` | what changed between two checkpoints, and which words did it |
| `ablate UNIT --ask "q"…` | zero a unit and compare answers (test a claim) |
| `png L1.mlp.w1` | a weight matrix as an image, rows grouped by what they write |
| `failures` / `edit "fact"` / `check` / `undo` | write weights by hand to fix a wrong fact, then verify all 703 |
| `jlens` / `jswap` / `jspace` | J-space prototype (after `python jspace.py`) |

Glyphs: weights `e d c b a . A B C D E` (strong − … ≈0 … strong +), heat `· ░ ▒ ▓ █`, time `▁▂▃▄▅▆▇█`.

## How the per-word attribution is exact

Each step, `torch.func` computes one gradient per trained answer word. The optimizer is RMSprop
without momentum, so the update `-lr · g / (sqrt(v / (1 - α^t)) + ε)` is linear in `g` and splits
into one piece per word with no approximation. Units are embedding rows, attention heads, MLP
neurons, and unembedding columns.

## How edits are written (no gradient descent)

A rank-1 change to the last MLP's output weights, in closed form (key → value, as in ROME):
the key is the hidden activation where the wrong word comes out, and the value is the residual
change that moves the logit from the wrong word to the right one. The default `protect` mode
holds the 128 most similar keys of other facts fixed, re-tests all facts, and rejects the edit if
anything broke.

| mode | fixed | broken | weights touched |
|---|---|---|---|
| sparse (8 neurons) | 12 | 62 | 2.0% |
| rome (covariance) | 12 | 18 | 6.2% |
| **protect + verify** | **10** | **0** | 6.1% |

Result on the trained model: 691 → 701 / 703 facts right, nothing else broken. The 2 rejected
edits (bee, spider) fail on the article: "a bee is **an** omnivore" means the wrong diet word was
already planned upstream, so fixing the article alone breaks every other animal sentence.

## hello-llm

Hand-written transformer: 11 layers, word + character tokens, 703 facts compiled into MLP neurons,
a relation classifier, a 2-hop layer (`is ottawa in europe?` → canada → north america), list
answers, yes/no checks, and copying for unknown words. On questions written by another model,
unseen before scoring: 94.0 → 92.7 → 98.3 → 88.7 → 83.2 → 96.5% across 6 rounds
(`heldout*.json`, `python eval.py heldout6.json`). `test.py` proves the answers are computed:
zeroing one fact neuron breaks every phrasing of that fact, and an audit finds 0 neurons mapping
question words straight to answer words.

## J-space (prototype)

`jspace.py` computes the Jacobian lens from Anthropic's
[A global workspace in language models](https://transformer-circuits.pub/2026/workspace/index.html):
`J_l = E[∂h_final,t' / ∂h_l,t]`. `wsv jspace` checks whether a workspace emerges: a swap along J-lens
directions beats random directions, and the J part is a small share of the activation. With 2 layers
the verdict is *not yet* (the France ↔ China swap flips 1 of 4 answers). Next is a 6-layer run.
See the LATER list in `jspace.py`.

## Credits

- Weight-space programming: Takehi, Teoh, Zhang, Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (2026)
- J-lens / J-space: Anthropic, *A global workspace in language models* (2026)
- Key-value edits: Meng et al., *Locating and Editing Factual Associations in GPT* (ROME, 2022)
- Facts and held-out questions: MiniMax-M3, cross-checked by hand (edits listed in `facts.json`)
