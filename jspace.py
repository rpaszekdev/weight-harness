"""Jacobian lens (J-lens) for the tiny trained model: a first look at whether a J-space emerges.

After Anthropic, "A global workspace in language models" (transformer-circuits.pub/2026/workspace,
2026-07-06). For each layer l, J_l is the average Jacobian from the residual stream entering block l
to the final residual stream, over source positions t, target positions t' >= t, and many contexts:

    J_l = E[ d h_final,t' / d h_l,t ]        lens(h) = softmax( LN_final(J_l h) @ W_U )

The sum over t' is taken inside one backward pass (causality makes every t' < t term zero), so each
context costs d_model backward passes. wsv.py reads the result: jlens (readout), jswap (the paper's
France -> China test), jspace (emergence report).

Run: python jspace.py                final checkpoint, 200 training sequences (needs run/ from train.py)
     python jspace.py --all-ckpts    every checkpoint: does a J-space appear during training?
Writes run/jlens@STEP.npz.

LATER (not in this prototype):
  - gradient-pursuit decomposition with k up to 25, as in the paper (wsv uses greedy pursuit, k=5)
  - J-space ablation: remove the top-k J coordinates at every position, see which skills survive
  - "think of a <category>" report test and thought-injection test
  - connectivity: how many weights read from / write to J directions vs random directions
  - retrain with more layers (python train.py --layers 6): with 2 layers there is no middle band
    of layers where a workspace could live, so a clean J-space is not expected here yet
"""
import argparse
import json
import os
import random

import numpy as np
import torch

import corpus
from train import RUN, Tiny, encode


def load_model(cfg, ckpts, step):
    model = Tiny(**cfg["model"])
    model.load_state_dict({n: torch.tensor(ckpts[f"{n}@{step}"]) for n, _ in model.named_parameters()})
    return model.eval()


@torch.no_grad()
def residual(model, ids, l):
    """The residual stream entering block l (l = 0: embeddings + positions)."""
    x = (model.wte[ids] + model.wpe[: len(ids)])[None]
    for blk in model.blocks[:l]:
        x = blk(x)
    return x[0]


def jacobian_sum(model, ids, l):
    """Sum over all (t, t' >= t) of d h_final,t' / d h_l,t for one context, and how many pairs that is."""
    h = residual(model, ids, l)

    def to_final(h):
        x = h[None]
        for blk in model.blocks[l:]:
            x = blk(x)
        return x[0].sum(0)                      # summing targets t'; causal attention zeroes t' < t

    jac = torch.func.jacrev(to_final)(h)        # [d_out, T, d_in]
    return jac.sum(1), len(ids) * (len(ids) + 1) // 2


def jlens_for(model, contexts, layers):
    out = {}
    for l in range(layers):
        total, pairs = 0, 0
        for ids in contexts:
            j, n = jacobian_sum(model, torch.tensor(ids), l)
            total, pairs = total + j, pairs + n
        out[f"J{l}"] = (total / pairs).detach().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--step", type=int, help="checkpoint (default: last)")
    ap.add_argument("--all-ckpts", action="store_true")
    ap.add_argument("--contexts", type=int, default=200, help="training sequences to average over")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    idx = {t: i for i, t in enumerate(cfg["vocab"])}
    exs = [ex for ex in corpus.examples() if not ex["held_out"]]
    random.Random(args.seed).shuffle(exs)
    contexts = [encode(ex, idx)[0] for ex in exs[: args.contexts]]
    steps = cfg["ckpt_steps"] if args.all_ckpts else [args.step if args.step is not None else cfg["ckpt_steps"][-1]]
    ckpts = np.load(os.path.join(args.run, "ckpts.npz"))
    for step in steps:
        model = load_model(cfg, ckpts, step)
        lens = jlens_for(model, contexts, cfg["model"]["layers"])
        path = os.path.join(args.run, f"jlens@{step}.npz")
        np.savez(path, contexts=len(contexts), **lens)
        print(f"step {step}: wrote {path}  ({len(lens)} layers × {cfg['model']['d']}² over {len(contexts)} contexts)")


if __name__ == "__main__":
    main()
