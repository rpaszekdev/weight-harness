"""Train a tiny GPT on hello-llm's facts and record which training word moved which weights.

Every step the gradient is split exactly into one piece per trained answer word (torch.func,
one gradient per target token). The optimizer is RMSprop without momentum, so each step's
update is  -lr * g / (sqrt(v / (1 - alpha^t)) + eps)  and splits into the same per-word pieces
with no approximation. Units = embedding rows, attention heads, MLP neurons, unembedding columns.

Run: python train.py                  3000 steps, CPU, 2 threads
     python train.py --quick          400 steps, a first look
Writes run/ (real parameters at every checkpoint + per-word pushes) for wsv.py to read.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

import corpus
from model import WEIGHTS, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
BOS, EOT = "^", "\n"
TOP_K = 20  # per word and checkpoint interval: how many most-pushed units to keep


def hello_vocab():
    """hello-llm's vocabulary; np.load is lazy, so only the vocab array leaves the big file."""
    with np.load(WEIGHTS) as z:
        vocab = [str(t) for t in z["vocab"]]
    return vocab, {t: i for i, t in enumerate(vocab)}


def label(tok):
    return {"\n": "<end>", "^": "<bos>", " ": "<spell>"}.get(tok, tok[1:] if tok.startswith(" ") else f"'{tok}'")


def prompt_ids(ex, idx):
    return [idx[BOS]] + tokenize({"index": idx}, ex["q"]) + [idx[EOT]]


def encode(ex, idx):
    """[bos] question [end] answer [end] -> token ids and the positions whose next token is trained."""
    prompt = prompt_ids(ex, idx)
    seq = prompt + [idx[t] for t in ex["answer"]] + [idx[EOT]]
    return seq, list(range(len(prompt) - 1, len(seq) - 1))


# ---------- model ----------
class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Parameter(torch.randn(d, d) * d ** -0.5) for _ in range(4))
        self.w1, self.b1 = nn.Parameter(torch.randn(d, ff) * d ** -0.5), nn.Parameter(torch.zeros(ff))
        self.w2, self.b2 = nn.Parameter(torch.randn(ff, d) * ff ** -0.5), nn.Parameter(torch.zeros(d))

    def forward(self, x):
        b, t, d = x.shape
        a = self.ln1(x)
        q, k, v = ((a @ w).reshape(b, t, self.heads, -1).transpose(1, 2) for w in (self.q, self.k, self.v))
        s = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
        x = x + (s.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d) @ self.o
        return x + torch.relu(self.ln2(x) @ self.w1 + self.b1) @ self.w2 + self.b2


class Tiny(nn.Module):
    def __init__(self, vocab, ctx, d, layers, heads, ff):
        super().__init__()
        self.wte = nn.Parameter(torch.randn(vocab, d) * 0.1)
        self.wpe = nn.Parameter(torch.randn(ctx, d) * 0.1)
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.wu = nn.Parameter(torch.randn(d, vocab) * d ** -0.5)

    def forward(self, idx):
        x = self.wte[idx] + self.wpe[: idx.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.lnf(x) @ self.wu


# ---------- units: which named piece of the model each weight belongs to ----------
def build_units(model, vocab, cfg):
    """Unit names + for every parameter a LongTensor (one unit id per element)."""
    names = []

    def new(n):
        names.append(n)
        return len(names) - 1

    d, ff, dh = cfg["d"], cfg["ff"], cfg["d"] // cfg["heads"]
    emb = torch.tensor([new(f"emb:{label(t)}") for t in vocab])
    pos = torch.tensor([new(f"pos:{p}") for p in range(cfg["ctx"])])
    per_param = {"wte": emb[:, None].expand(-1, d), "wpe": pos[:, None].expand(-1, d)}
    for l in range(cfg["layers"]):
        heads = torch.tensor([new(f"L{l}.attn.h{h}") for h in range(cfg["heads"])])
        neurons = torch.tensor([new(f"L{l}.mlp.n{i}") for i in range(ff)])
        misc = new(f"L{l}.misc")
        col_head = heads[torch.arange(d) // dh]
        for n in "qkv":
            per_param[f"blocks.{l}.{n}"] = col_head[None, :].expand(d, -1)
        per_param[f"blocks.{l}.o"] = col_head[:, None].expand(-1, d)
        per_param[f"blocks.{l}.w1"] = neurons[None, :].expand(d, -1)
        per_param[f"blocks.{l}.b1"] = neurons
        per_param[f"blocks.{l}.w2"] = neurons[:, None].expand(-1, d)
        for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "b2"):
            per_param[f"blocks.{l}.{n}"] = torch.full((d,), misc)
    lnf = new("lnf")
    per_param["lnf.weight"] = per_param["lnf.bias"] = torch.full((d,), lnf)
    per_param["wu"] = torch.tensor([new(f"unemb:{label(t)}") for t in vocab])[None, :].expand(d, -1)
    params = dict(model.named_parameters())
    assert set(per_param) == set(params), set(params) ^ set(per_param)
    assert all(per_param[k].shape == params[k].shape for k in params)
    return names, {k: v.reshape(-1).contiguous() for k, v in per_param.items()}


# ---------- one training step with exact per-word updates ----------
def per_token_grads(model):
    def loss(params, seq, mask):  # seq [T+1], mask [T]: 1 at the one position this item trains
        logits = functional_call(model, params, (seq[:-1].unsqueeze(0),))[0]
        return (F.cross_entropy(logits, seq[1:], reduction="none") * mask).sum()
    return vmap(grad_and_value(loss), in_dims=(None, 0, 0))


def make_items(batch, eot):
    """One item per trained token: padded sequence, one-hot mask of its position, the target word id."""
    t = max(len(s) for s, _ in batch)
    seqs, masks, targets = [], [], []
    for seq, positions in batch:
        padded = seq + [eot] * (t - len(seq))
        for p in positions:
            mask = [0.0] * (t - 1)
            mask[p] = 1.0
            seqs.append(padded)
            masks.append(mask)
            targets.append(seq[p + 1])
    return torch.tensor(seqs), torch.tensor(masks), torch.tensor(targets)


def train_step(model, grads_fn, units, state, items, push, args):
    """RMSprop step; adds each word's exact share of the update (per unit, L2) into push[word]."""
    seqs, masks, targets = items
    n = len(targets)
    g_tok, losses = grads_fn({k: v.detach() for k, v in model.named_parameters()}, seqs, masks)
    sq = torch.zeros(n, push.shape[1])
    state["_t"] = state.get("_t", 0) + 1
    debias = 1 - args.alpha ** state["_t"]                     # v starts at 0: undo that bias (still linear in g)
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = g_tok[name].mean(0)                            # the batch gradient
            v = state.setdefault(name, torch.zeros_like(p))
            v.mul_(args.alpha).add_((1 - args.alpha) * g * g)
            scale = args.lr / ((v / debias).sqrt() + 1e-8)
            p -= scale * g
            piece = (scale * g_tok[name] / n).reshape(n, -1)   # word i's share; the n pieces sum to the update
            sq.index_add_(1, units[name], piece * piece)
        push.index_add_(0, targets, sq.sqrt())
    return float(losses.mean())


# ---------- probes: fixed questions, to see when each fact was learned ----------
def probe_set(exs, idx):
    """One trained question per fact (pattern 0), teacher-forced up to the object word."""
    probes = []
    for ex in exs:
        if ex["pattern"] == 0:
            seq, positions = encode(ex, idx)
            obj = idx[" " + ex["o"]]
            p = next(p for p in positions if seq[p + 1] == obj)
            probes.append({"fact": ex["fact"], "q": ex["q"], "ids": seq[:p + 1], "obj": obj})
    return probes


@torch.no_grad()
def probe_loss(model, probes):
    return [float(F.cross_entropy(model(torch.tensor([pr["ids"]]))[0, -1:], torch.tensor([pr["obj"]])))
            for pr in probes]


@torch.no_grad()
def generate(model, ids, eot, max_new=12):
    out = []
    for _ in range(min(max_new, model.wpe.shape[0] - len(ids))):
        t = int(model(torch.tensor([ids + out]))[0, -1].argmax())
        if t == eot:
            break
        out.append(t)
    return out


def heldout_accuracy(model, exs, idx):
    held = [ex for ex in exs if ex["held_out"]]
    ok = sum(generate(model, prompt_ids(ex, idx), idx[EOT]) == [idx[t] for t in ex["answer"]] for ex in held)
    return ok, len(held)


def top_k(delta, step):
    """Sparse record of one checkpoint interval: for each word, its TOP_K most-pushed units."""
    rows = torch.nonzero(delta.sum(1) > 0).flatten()
    vals, cols = torch.topk(delta[rows], min(TOP_K, delta.shape[1]), dim=1)
    words = rows[:, None].expand_as(cols)
    return [np.full(cols.numel(), step), words.flatten().numpy(), cols.flatten().numpy(), vals.flatten().numpy()]


# ---------- main ----------
def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="400 steps, checkpoint every 50")
    ap.add_argument("--batch", type=int, default=8, help="questions per step")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.99, help="RMSprop decay")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (keep low to keep the machine usable)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RUN)
    args = ap.parse_args()
    if args.quick:
        args.steps, args.ckpt_every = 400, 50
    return args


def save(out, cfg, vocab, names, units, exs, probes, log):
    os.makedirs(out, exist_ok=True)
    facts = [(rel, s, o) for rel, (_, s), (_, o) in corpus.load_facts()]
    meta = {"model": cfg, "vocab": vocab, "units": names, "facts": facts, "probes": probes,
            "ckpt_steps": log["steps"], "train": log["args"], "heldout": log["heldout"], "seconds": log["seconds"],
            "train_loss": log["train_loss"], "examples": len(exs)}
    json.dump(meta, open(os.path.join(out, "config.json"), "w"))
    np.savez_compressed(os.path.join(out, "ckpts.npz"), **log["ckpts"])
    np.savez_compressed(os.path.join(out, "unit_index.npz"), **{k: v.numpy() for k, v in units.items()})
    np.save(os.path.join(out, "push.npy"), log["push"].numpy().astype(np.float32))
    np.save(os.path.join(out, "seen.npy"), log["seen"].numpy())
    np.save(os.path.join(out, "loss.npy"), np.array(log["losses"], dtype=np.float32))
    parts = list(zip(*log["intervals"])) if log["intervals"] else [[]] * 4
    np.savez_compressed(os.path.join(out, "intervals.npz"),
                        **{k: np.concatenate(p) if len(p) else np.array([]) for k, p in zip(("step", "word", "unit", "value"), parts)})


def main():
    args = parse()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    vocab, idx = hello_vocab()
    exs = corpus.examples()
    train = [encode(ex, idx) for ex in exs if not ex["held_out"]]
    ctx = max(len(encode(ex, idx)[0]) for ex in exs) + 1
    cfg = dict(vocab=len(vocab), ctx=ctx, d=args.d, layers=args.layers, heads=args.heads, ff=args.ff)
    model = Tiny(**cfg)
    names, units = build_units(model, vocab, cfg)
    grads_fn, probes, state = per_token_grads(model), probe_set(exs, idx), {}
    push, seen = torch.zeros(len(vocab), len(names)), torch.zeros(len(vocab))
    log = {"steps": [], "losses": [], "intervals": [], "ckpts": {}, "train_loss": [], "args": vars(args)}
    last, t0, loss = push.clone(), time.time(), float("nan")
    print(f"{len(train)} training questions, {len(names)} units, "
          f"{sum(p.numel() for p in model.parameters()):,} params, {args.steps} steps")
    for step in range(args.steps + 1):
        if step % args.ckpt_every == 0 or step == args.steps:
            log["steps"].append(step)
            log["losses"].append(probe_loss(model, probes))
            log["ckpts"].update({f"{name}@{step}": p.detach().numpy().copy() for name, p in model.named_parameters()})
            if step:
                log["intervals"].append(top_k(push - last, step))
                last = push.clone()
            rate = (time.time() - t0) / max(step, 1)
            print(f"step {step:5d}  train loss {loss:.3f}  probe loss {np.mean(log['losses'][-1]):.3f}  "
                  f"{rate:.2f}s/step  eta {rate * (args.steps - step) / 60:.1f} min", flush=True)
        if step == args.steps:
            break
        items = make_items(random.sample(train, args.batch), idx[EOT])
        seen.index_add_(0, items[2], torch.ones(len(items[2])))
        loss = train_step(model, grads_fn, units, state, items, push, args)
        log["train_loss"].append(loss)
    ok, n = heldout_accuracy(model, exs, idx)
    print(f"held-out wording: {ok}/{n} answers exactly right")
    log.update(push=push, seen=seen, heldout={"correct": ok, "total": n}, seconds=time.time() - t0)
    save(args.out, cfg, vocab, names, units, exs, probes, log)
    print(f"wrote {args.out}/  ->  python wsv.py atlas")


if __name__ == "__main__":
    main()
