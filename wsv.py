"""wsv: weight-space viewer. Look inside the tiny trained model as text an LLM can read (numpy only).

  python wsv.py atlas  [--budget 400]           the whole model on one screen
  python wsv.py ask    "capital of peru?"       the trained model's answer
  python wsv.py trace  "capital of peru?"       which heads/neurons produced the answer word
  python wsv.py neuron L1.mlp.n41               word lens, raw weights, who built it, when, answer key
  python wsv.py head   L0.attn.h1               where it looks, whether it copies, who built it
  python wsv.py word   lima                     which units this training word moved (heat map)
  python wsv.py diff   500 700                  what changed between two checkpoints, and which words did it
  python wsv.py timeline                        training as a graph: facts learned, loss, what moved, who moved it
  python wsv.py ablate L1.mlp.n41 --ask "capital of peru?" "capital of chile?"
  python wsv.py png    L1.mlp.w1 [--out f.png]  a weight matrix as an image, rows sorted by top word
  editing (wsedit.py): write weights by hand, closed form, no gradient descent; --base ignores edits
  python wsv.py failures                        facts the model gets wrong, and the first wrong word
  python wsv.py edit   "peru capital lima"      rank-1 weight edit so this fact comes out right
  python wsv.py check                           all facts vs the base model: fixed / broken, weights touched
  python wsv.py edits | undo [ID]               list / remove edits
  J-space prototype (after python jspace.py; method from transformer-circuits.pub/2026/workspace):
  python wsv.py jlens  "currency of france?"    silent words per position and layer (J-lens vs logit lens)
  python wsv.py jswap  france china [--layer 1] the paper's France→China swap on our 4 relations
  python wsv.py jspace [--pairs 30]             does a J-space emerge? causal + small, per layer

Reads run/ from train.py: the real parameters at every checkpoint and the exact per-word updates.
Add --step N to look at an earlier checkpoint. Glyphs:
  weights  e d c b a . A B C D E   (strong − … ≈0 … strong +)    heat  · ░ ▒ ▓ █    time  ▁▂▃▄▅▆▇█
"""
import argparse
import json
import os
import re
import struct
import zlib

import numpy as np

import corpus
from model import WEIGHTS, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run")
BOS, EOT = "^", "\n"
NEG, POSG, HEAT, SPARK = "abcde", "ABCDE", "·░▒▓█", "▁▂▃▄▅▆▇█"
LEARNED = 0.5  # probe loss below this = the fact counts as learned


# ---------- loading ----------
def label(tok):
    return {"\n": "<end>", "^": "<bos>", " ": "<spell>"}.get(tok, tok[1:] if tok.startswith(" ") else f"'{tok}'")


class Run:
    """Everything train.py wrote. Parameters are the real ones at the chosen checkpoint."""

    def __init__(self, path=RUN, step=None, edits=True):
        if not os.path.exists(os.path.join(path, "config.json")):
            raise SystemExit(f"no {path}/config.json yet: run python train.py first")
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        self.m, self.vocab, self.units = self.cfg["model"], self.cfg["vocab"], self.cfg["units"]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.unit_id = {n: i for i, n in enumerate(self.units)}
        self.steps = self.cfg["ckpt_steps"]
        self.ckpts = np.load(os.path.join(path, "ckpts.npz"))
        self.names = sorted({k.split("@")[0] for k in self.ckpts.files})
        self.step = max(s for s in self.steps if s <= (self.steps[-1] if step is None else step))
        self.p = self.params(self.step)
        self.edits = load_edits(path) if edits and self.step == self.steps[-1] else []
        for e in self.edits:  # hand-written weight edits from wsedit.py, applied on top of the final checkpoint
            with np.load(os.path.join(path, "edits", f"{e['id']}.npz")) as z:
                self.p = {**self.p, **{k: self.p[k] + z[k] for k in z.files}}
        self.push = np.load(os.path.join(path, "push.npy"))
        self.seen = np.load(os.path.join(path, "seen.npy"))
        self.loss = np.load(os.path.join(path, "loss.npy"))
        with np.load(os.path.join(path, "unit_index.npz")) as z:
            self.unit_index = {k: z[k] for k in z.files}
        self.types = {}
        for _, (ts, s), (to, o) in corpus.load_facts():
            self.types[s], self.types[o] = ts, to

    def params(self, step):
        return {n: self.ckpts[f"{n}@{step}"] for n in self.names}

    def word_id(self, word):
        for tok in (" " + word, word):
            if tok in self.index:
                return self.index[tok]
        raise SystemExit(f"'{word}' is not in the vocabulary")

    def kind(self, tok_id):
        return self.types.get(label(self.vocab[tok_id]), "word")


def load_edits(path):
    index = os.path.join(path, "edits", "index.json")
    return json.load(open(index)) if os.path.exists(index) else []


def blk(p, l, name):
    return p[f"blocks.{l}.{name}"]


# ---------- the model, same math as train.Tiny ----------
def ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5) * w + b


def forward(run, ids, cut=(), p=None, edit=None):
    """ids -> logits [T, V], plus every head's output and every neuron's activation (for trace/lens).
    edit = (l, fn): replace the residual stream entering block l (l = layers: before the final norm) by fn(x)."""
    p, m, t = p or run.p, run.m, len(ids)
    d, h = m["d"], m["heads"]
    x = p["wte"][ids] + p["wpe"][:t]
    rec = {"emb": x.copy(), "layers": []}
    causal = np.triu(np.full((t, t), -np.inf), 1)
    for l in range(m["layers"]):
        if edit and edit[0] == l:
            x = edit[1](x)
        rec.setdefault("resid", []).append(x)
        a = ln(x, blk(p, l, "ln1.weight"), blk(p, l, "ln1.bias"))
        q, k, v = ((a @ blk(p, l, n)).reshape(t, h, d // h).transpose(1, 0, 2) for n in "qkv")
        s = q @ k.transpose(0, 2, 1) / np.sqrt(d // h) + causal
        att = np.exp(s - s.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        mix = att @ v
        mix[[i for i in range(h) if f"L{l}.attn.h{i}" in cut]] = 0
        head_out = np.einsum("htd,hde->hte", mix, blk(p, l, "o").reshape(h, d // h, d))
        x = x + head_out.sum(0)
        hid = np.maximum(0, ln(x, blk(p, l, "ln2.weight"), blk(p, l, "ln2.bias")) @ blk(p, l, "w1") + blk(p, l, "b1"))
        hid[:, [int(c.split(".n")[1]) for c in cut if c.startswith(f"L{l}.mlp.n")]] = 0
        x = x + hid @ blk(p, l, "w2") + blk(p, l, "b2")
        rec["layers"].append({"att": att, "head_out": head_out, "hid": hid})
    if edit and edit[0] == m["layers"]:
        x = edit[1](x)
    rec["x"] = x
    return ln(x, p["lnf.weight"], p["lnf.bias"]) @ p["wu"], rec


def prompt(run, q):
    return [run.index[BOS]] + tokenize({"index": run.index}, q) + [run.index[EOT]]


def answer(run, q, cut=(), edit=None):
    ids, out = prompt(run, q), []
    while len(out) < 12 and len(ids) + len(out) < run.m["ctx"]:
        t = int(forward(run, ids + out, cut, edit=edit)[0][-1].argmax())
        if run.vocab[t] == EOT:
            break
        out.append(t)
    return ids, out


def text(run, toks):
    return "".join(run.vocab[t] for t in toks).strip()


# ---------- lenses: weights read as words ----------
def word_scores(run, vec):
    """What a residual-stream direction votes for at the output (final LayerNorm scale, then unembedding)."""
    return ((vec - vec.mean()) * run.p["lnf.weight"]) @ run.p["wu"]


def read_scores(run, l, col, norm="ln2"):
    """How much each word, as layer l sees it (LayerNorm'd embedding), drives an input weight column."""
    e = ln(run.p["wte"], blk(run.p, l, f"{norm}.weight"), blk(run.p, l, f"{norm}.bias"))
    return e @ col


def top_words(run, scores, k=5, sign=1):
    order = np.argsort(-sign * scores)[:k]
    return " · ".join(f"{label(run.vocab[i])} {scores[i]:+.2f}" for i in order)


def contributions(run, rec, pos, tok):
    """Direct logit attribution: every component's exact share of tok's logit at pos (sums to the logit − bias)."""
    x = rec["x"][pos]
    col = run.p["lnf.weight"] * run.p["wu"][:, tok] / np.sqrt(x.var() + 1e-5)

    def c(vec):
        return float((vec - vec.mean()) @ col)

    out = [("emb+pos", c(rec["emb"][pos]))]
    for l, layer in enumerate(rec["layers"]):
        out += [(f"L{l}.attn.h{h}", c(layer["head_out"][h, pos])) for h in range(run.m["heads"])]
        writes = layer["hid"][pos][:, None] * blk(run.p, l, "w2")
        vals = (writes - writes.mean(1, keepdims=True)) @ col
        out += [(f"L{l}.mlp.n{i}", float(vals[i])) for i in np.flatnonzero(layer["hid"][pos] > 0)]
        out.append((f"L{l}.b2", c(blk(run.p, l, "b2"))))
    return sorted(out, key=lambda kv: -kv[1])


# ---------- glyphs ----------
def glyphs(w):
    m = float(np.abs(w).max()) or 1.0
    lv = np.minimum(4, (np.abs(w) / m * 5).astype(int))
    return "".join("." if abs(x) < 0.1 * m else (NEG if x < 0 else POSG)[i] for x, i in zip(w, lv))


def heat(vals):
    m = float(np.max(vals)) or 1.0
    return "".join(HEAT[min(4, int(v / m * 4.999))] if v > 0.02 * m else HEAT[0] for v in vals)


def spark(vals):
    vals = np.asarray(vals, dtype=float)
    lo, hi = vals.min(), vals.max()
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-12) * 7)] for v in vals)


def shares(row, k=4, names=None):
    total = float(row.sum()) or 1.0
    order = np.argsort(-row)[:k]
    return " · ".join(f"{names[i]} {100 * row[i] / total:.0f}%" for i in order if row[i] > 0)


def unit_norms(run, p):
    tot = np.zeros(len(run.units))
    for name, ids in run.unit_index.items():
        tot += np.bincount(ids, weights=p[name].astype(float).ravel() ** 2, minlength=len(run.units))
    return np.sqrt(tot)


def probe_acts(run):
    """Every MLP neuron's activation where each fact's object word is predicted: [probe, layer, neuron]."""
    tag = f"+e{len(run.edits)}-{run.edits[-1]['id']}" if run.edits else ""
    cache = os.path.join(run.path, f"acts@{run.step}{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    acts = np.stack([np.stack([layer["hid"][-1] for layer in forward(run, pr["ids"])[1]["layers"]])
                     for pr in run.cfg["probes"]])
    np.save(cache, acts)
    return acts


def fact_name(run, f):
    rel, s, o = run.cfg["facts"][f]
    return f"{s} {rel} {o}"


def learned_at(run, probe_i):
    hits = [s for s, row in zip(run.steps, run.loss) if row[probe_i] < LEARNED]
    return hits[0] if hits else None


# ---------- answer key: the hand-written hello-llm ----------
def hello_neuron(run, fact_i):
    """hello-llm's fact neuron for the same fact (real weights from weights.npz) and the dims it reads."""
    rel, s, o = run.cfg["facts"][fact_i]
    if not os.path.exists(WEIGHTS):
        return None
    with np.load(WEIGHTS) as z:
        names, layer = [str(n) for n in z["dims"]], int(z["fact_layer"])
        w1, w2 = z[f"w1{layer}"], z[f"w2{layer}"]
    col = {n: i for i, n in enumerate(names)}
    js, jo = col.get(f"subj:{(run.types[s], s)}"), col.get(f"obj:{(run.types[o], o)}")
    if js is None or jo is None:
        return None
    hits = np.flatnonzero((w2[:, js] != 0) & (w2[:, jo] != 0))
    return (layer, int(hits[0]), [names[i] for i in np.flatnonzero(w1[:, hits[0]])]) if len(hits) else None


# ---------- commands ----------
def parse_unit(name):
    """'L1.mlp.n41' -> (1, 'mlp', 41);  'L0.attn.h2' -> (0, 'attn', 2)"""
    try:
        l, kind, i = name.split(".")
        return int(l[1:]), kind, int(i[1:])
    except ValueError:
        raise SystemExit(f"unit names look like L1.mlp.n41 or L0.attn.h2, not '{name}'")


def words_of(run):
    return [label(t) for t in run.vocab]


def head_looks(run, probes):
    """Where every head looks from the position that predicts the answer word, averaged over probes."""
    tags = ["subject", "relation word", "<bos>", "question end", "answer so far", "other"]
    tally = {(l, h): dict.fromkeys(tags, 0.0) for l in range(run.m["layers"]) for h in range(run.m["heads"])}
    for pr in probes:
        ids = pr["ids"]
        rel, s, _ = run.cfg["facts"][pr["fact"]]
        end = ids.index(run.index[EOT])
        kinds = [("<bos>" if i == 0 else "question end" if i == end else "answer so far" if i > end
                  else "subject" if label(run.vocab[t]) == s else "relation word" if label(run.vocab[t]) == rel
                  else "other") for i, t in enumerate(ids)]
        for l, layer in enumerate(forward(run, ids)[1]["layers"]):
            for h in range(run.m["heads"]):
                for kind, a in zip(kinds, layer["att"][h, -1]):
                    tally[(l, h)][kind] += a / len(probes)
    return tally


def cmd_atlas(run, budget):
    probes = run.cfg["probes"][::7]
    learned = int((run.loss[run.steps.index(run.step)] < LEARNED).sum())
    held = run.cfg["heldout"]
    lines = [f"atlas  step {run.step} · {sum(v.size for v in run.p.values()):,} params · "
             f"{learned}/{len(run.cfg['probes'])} facts learned · held-out wording {held['correct']}/{held['total']}"]
    e = run.p["wte"] / (np.linalg.norm(run.p["wte"], axis=1, keepdims=True) + 1e-9)
    ent = [i for i in range(len(run.vocab)) if run.kind(i) != "word"]
    sim = e[ent] @ e.T
    sim[np.arange(len(ent)), ent] = -9
    purity = np.mean([run.kind(int(sim[j].argmax())) == run.kind(i) for j, i in enumerate(ent)])
    lines.append(f"  emb    {len(run.vocab)}×{run.m['d']}  nearest neighbour has the same type for {purity:.0%} of entity words")
    tally = head_looks(run, probes)
    for l in range(run.m["layers"]):
        looks = [max(tally[(l, h)].items(), key=lambda kv: kv[1]) for h in range(run.m["heads"])]
        lines.append(f"  L{l}.attn  " + " · ".join(f"h{h} → {k} {v:.0%}" for h, (k, v) in enumerate(looks)))
        w1, w2 = blk(run.p, l, "w1"), blk(run.p, l, "w2")
        strength = np.linalg.norm(w1, axis=0) * np.linalg.norm(w2, axis=1)
        strong = np.argsort(-strength)[: run.m["ff"] // 4]
        kinds = {}
        for i in strong:
            k = run.kind(int(word_scores(run, w2[i]).argmax()))
            kinds[k] = kinds.get(k, 0) + 1
        lines.append(f"  L{l}.mlp   strongest {len(strong)} write: " +
                     " · ".join(f"{k} {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
    lines.append("  output   the answer word's logit comes from: " + output_sources(run, probes))
    emit(lines, budget)


def output_sources(run, probes):
    groups = {}
    for pr in probes:
        rec = forward(run, pr["ids"])[1]
        for name, v in contributions(run, rec, len(pr["ids"]) - 1, pr["obj"]):
            g = name.rsplit(".", 1)[0] if name.startswith("L") else name
            groups[g] = groups.get(g, 0.0) + max(v, 0.0)
    total = sum(groups.values()) or 1.0
    return " · ".join(f"{g} {100 * v / total:.0f}%" for g, v in sorted(groups.items(), key=lambda kv: -kv[1]))


def cmd_trace(run, q, top):
    ids, out = answer(run, q)
    if not out:
        return print(f"{q!r} -> (no answer)")
    ents = [i for i, t in enumerate(out) if run.kind(t) != "word"]   # the answer is the last entity (frames end on it)
    at = ents[-1] if ents else 0
    key, pos = out[at], len(ids) + at - 1
    logits, rec = forward(run, ids + out)
    prob = np.exp(logits[pos] - logits[pos].max())
    print(f"{q!r} -> {text(run, out)!r}")
    print(f"  key word '{label(run.vocab[key])}' (p {prob[key] / prob.sum():.2f}) is built by:")
    for name, v in contributions(run, rec, pos, key)[:top]:
        print(f"    {name:14s} {v:+6.2f}  {describe(run, name, rec, pos, ids + out)}")


def describe(run, name, rec, pos, ids):
    if ".mlp.n" in name:
        l, _, i = parse_unit(name)
        return "reads " + top_words(run, read_scores(run, l, blk(run.p, l, "w1")[:, i]), 3)
    if ".attn.h" in name:
        l, _, h = parse_unit(name)
        att = rec["layers"][l]["att"][h, pos]
        return "looks at " + " · ".join(f"{label(run.vocab[ids[j]])} {att[j]:.0%}" for j in np.argsort(-att)[:3])
    return ""


def cmd_neuron(run, name):
    l, kind, i = parse_unit(name)
    if kind != "mlp":
        raise SystemExit("neuron takes an MLP unit like L1.mlp.n41 (use head for attention)")
    w1, w2 = blk(run.p, l, "w1")[:, i], blk(run.p, l, "w2")[i]
    reads, writes = read_scores(run, l, w1), word_scores(run, w2)
    acts = probe_acts(run)[:, l, i]
    top = np.argsort(-acts)[:3]
    uid = run.unit_id[name]
    norms = [unit_norms(run, run.params(s))[uid] for s in run.steps]
    print(f"{name}  (step {run.step}; lenses go through LayerNorm'd embeddings / the unembedding)")
    print(f"  reads     {top_words(run, reads)}")
    print(f"  against   {top_words(run, reads, sign=-1)}")
    print(f"  writes    {top_words(run, writes)}")
    print(f"  raw in    {glyphs(w1)}")
    print(f"  raw out   {glyphs(w2)}")
    print("  fires on  " + " · ".join(f"{fact_name(run, run.cfg['probes'][j]['fact'])} {acts[j]:.2f}" for j in top))
    print(f"  built by  {shares(run.push[:, uid], names=words_of(run))}   (exact share of all its updates)")
    print(f"  over time {spark(norms)}  |w| at steps {run.steps[0]}..{run.steps[-1]}; "
          f"its top fact learned at step {learned_at(run, int(top[0]))}")
    print(answer_key_line(run, run.cfg["probes"][int(top[0])]["fact"], reads, writes))


def answer_key_line(run, fact_i, reads, writes):
    hk = hello_neuron(run, fact_i)
    if not hk:
        return "  answer key (no hand-written hello-llm neuron for this fact)"
    rel, s, o = run.cfg["facts"][fact_i]
    top_r = {label(run.vocab[j]) for j in np.argsort(-reads)[:10]}
    top_w = {label(run.vocab[j]) for j in np.argsort(-writes)[:10]}
    marks = [f"{what} {'✓' if hit else '✗'}" for what, hit in
             (("subject", s in top_r), ("relation word", rel in top_r), ("object", o in top_w))]
    layer, j, dims = hk
    return f"  answer key hello-llm L{layer}#{j} reads {', '.join(dims)}  →  this neuron: {' · '.join(marks)}"


def cmd_head(run, name):
    l, kind, h = parse_unit(name)
    if kind != "attn":
        raise SystemExit("head takes an attention unit like L0.attn.h2")
    dh = run.m["d"] // run.m["heads"]
    looks = head_looks(run, run.cfg["probes"][::7])[(l, h)]
    e = ln(run.p["wte"], blk(run.p, l, "ln1.weight"), blk(run.p, l, "ln1.bias"))
    ov = e @ blk(run.p, l, "v")[:, h * dh:(h + 1) * dh] @ blk(run.p, l, "o")[h * dh:(h + 1) * dh]
    out = ((ov - ov.mean(1, keepdims=True)) * run.p["lnf.weight"]) @ run.p["wu"]
    ent = [i for i in range(len(run.vocab)) if run.kind(i) != "word"]
    copies = np.mean([int(out[i].argmax()) == i for i in ent])
    print(f"{name}  (step {run.step})")
    print("  looks at  " + " · ".join(f"{k} {v:.0%}" for k, v in sorted(looks.items(), key=lambda kv: -kv[1]) if v > 0.01))
    print(f"  copies    {copies:.0%} of entity words map onto themselves through its value/output weights")
    print(f"  built by  {shares(run.push[:, run.unit_id[name]], names=words_of(run))}")


def unit_groups(run):
    groups = [("emb", "emb:")] + [(f"L{l}.{k}", f"L{l}.{k}.") for l in range(run.m["layers"]) for k in ("attn", "mlp")]
    return groups + [("unemb", "unemb:")]


def cmd_word(run, w):
    wid = run.word_id(w)
    row = run.push[wid]
    print(f"{label(run.vocab[wid])}  trained on {int(run.seen[wid])}× · total push {row.sum():.3f}")
    for gname, prefix in unit_groups(run):
        ids = [i for i, n in enumerate(run.units) if n.startswith(prefix)]
        vals = row[ids]
        pooled = vals[: len(vals) // 64 * 64].reshape(64, -1).max(1) if len(vals) > 64 else vals
        best = ids[int(vals.argmax())]
        print(f"  {gname:9s} {heat(pooled)}  top {run.units[best]} ({100 * row[best] / (row.sum() or 1):.0f}%)")
    print(f"  top units {shares(row, 6, names=run.units)}")
    for j, pr in enumerate(run.cfg["probes"]):
        rel, s, o = run.cfg["facts"][pr["fact"]]
        if label(run.vocab[wid]) in (s, o):
            print(f"  fact {s} {rel} {o:14s} probe loss {spark(run.loss[:, j])}  learned at step {learned_at(run, j)}")


def cmd_diff(run, a, b, k=10):
    sa, sb = (max(s for s in run.steps if s <= x) for x in (a, b))
    pa, pb = run.params(sa), run.params(sb)
    delta = unit_norms(run, {n: pb[n] - pa[n] for n in run.names})
    with np.load(os.path.join(run.path, "intervals.npz")) as iv:
        sel = (iv["step"] > sa) & (iv["step"] <= sb)
        word, unit, value = iv["word"][sel], iv["unit"][sel], iv["value"][sel]
    print(f"diff step {sa} → {sb}   unit   Δ|w|   moved by (this interval)")
    for u in np.argsort(-delta)[:k]:
        m = unit == u
        by = np.bincount(word[m], weights=value[m], minlength=len(run.vocab))
        print(f"  {run.units[u]:16s} {delta[u]:.3f}  {shares(by, 3, names=words_of(run))}")


def cmd_ablate(run, name, questions):
    if name not in run.unit_id:
        raise SystemExit(f"no unit '{name}'")
    for q in questions:
        before, after = text(run, answer(run, q)[1]), text(run, answer(run, q, cut={name})[1])
        print(f"  {q!r}\n    before {before!r}\n    after  {after!r}   {'✓ same' if before == after else '✗ changed'}")


def matrix(run, name):
    """A named weight matrix as rows to draw, plus a label per row."""
    if name in ("emb", "unemb"):
        m = run.p["wte"] if name == "emb" else run.p["wu"].T
        return m, words_of(run), [run.kind(i) for i in range(len(run.vocab))]
    l, kind, part = name.split(".")
    l = int(l[1:])
    w = blk(run.p, l, part)
    if part in ("w1", "w2"):
        rows = w.T if part == "w1" else w
        tops = [int(word_scores(run, blk(run.p, l, "w2")[i]).argmax()) for i in range(rows.shape[0])]
        return rows, [f"n{i}→{label(run.vocab[t])}" for i, t in enumerate(tops)], [run.kind(t) for t in tops]
    return w, [f"dim{i}" for i in range(w.shape[0])], ["dim"] * w.shape[0]


def write_png(path, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def cmd_png(run, name, out, scale=4):
    m, labels, kinds = matrix(run, name)
    order = sorted(range(len(labels)), key=lambda i: (kinds[i], labels[i]))
    m = m[order]
    a = np.abs(m) / (np.abs(m).max() or 1.0)
    rgb = np.where(m[..., None] < 0, np.array([40, 110, 255]), np.array([255, 150, 20])) * a[..., None]
    rgb = np.repeat(np.repeat(rgb.astype(np.uint8), scale, 0), scale, 1)
    path = out or os.path.join(run.path, f"{name}@{run.step}.png")
    write_png(path, rgb)
    print(f"wrote {path}  ({m.shape[0]} rows × {m.shape[1]} cols, {scale}px each; blue −, orange +)")
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or kinds[order[i]] != kinds[order[start]]:
            print(f"  rows {start:4d}-{i - 1:4d}  {kinds[order[start]]:9s} e.g. {', '.join(labels[j] for j in order[start:min(i, start + 4)])}")
            start = i


def cmd_timeline(run, width=24):
    """How training went: facts learned, probe loss, which parts of the model moved, and the words that moved them."""
    groups = unit_groups(run)
    member = np.full(len(run.units), -1)
    for g, (_, prefix) in enumerate(groups):
        member[[i for i, n in enumerate(run.units) if n.startswith(prefix)]] = g
    with np.load(os.path.join(run.path, "intervals.npz")) as iv:
        iv = {k: iv[k] for k in iv.files}
    moved, prev = [], None
    for st in run.steps:
        p = run.params(st)
        d = unit_norms(run, {n: p[n] - prev[n] for n in run.names}) if prev else np.zeros(len(run.units))
        moved.append([np.sqrt((d[member == g] ** 2).sum()) for g in range(len(groups))])
        prev = p
    moved = np.array(moved) / (np.array(moved).max(0) + 1e-12)
    n = len(run.cfg["probes"])
    print(f"timeline  {run.steps[0]} → {run.steps[-1]}  ({len(run.steps)} checkpoints; heat = how much each part moved "
          f"since the previous checkpoint, scaled per part)")
    print(f"  {'step':>5}  {'facts learned':{width + 9}}  loss  " + " ".join(f"{g:>7}" for g, _ in groups) + "   words that moved it most")
    for k, st in enumerate(run.steps):
        learned = int((run.loss[k] < LEARNED).sum())
        bar = "█" * round(width * learned / n)
        sel = iv["step"] == st
        by = np.bincount(iv["word"][sel].astype(int), weights=iv["value"][sel], minlength=len(run.vocab)) if sel.any() else None
        top = shares(by, 3, names=words_of(run)) if by is not None else ""
        cells = " ".join(f"{HEAT[min(4, int(v * 4.999))] if v > 0.02 else HEAT[0]:>7}" for v in moved[k])
        print(f"  {st:5d}  {bar:{width}} {learned:3d}/{n}  {run.loss[k].mean():4.2f}  {cells}   {top}")
    marks = []
    for frac in (0.1, 0.5, 0.9):
        hit = [st for k, st in enumerate(run.steps) if (run.loss[k] < LEARNED).mean() >= frac]
        marks.append(f"{frac:.0%} of facts by step {hit[0] if hit else 'never'}")
    print("  " + " · ".join(marks))


# ---------- J-space prototype (jspace.py computes the Jacobians; see its LATER list) ----------
def jlens_mats(run):
    path = os.path.join(run.path, f"jlens@{run.step}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}: run python jspace.py --step {run.step}")
    with np.load(path) as z:
        return [z[f"J{l}"] for l in range(run.m["layers"])]


def jvectors(run, j):
    """J-lens vectors [V, d]: each word's direction at this layer (through J and the final norm's centering/scale)."""
    w = run.p["wu"] * run.p["lnf.weight"][:, None]
    return (j.T @ (w - w.mean(0, keepdims=True))).T


def lens_read(run, j, h, k=3):
    s = ln(h @ j.T, run.p["lnf.weight"], run.p["lnf.bias"]) @ run.p["wu"]
    return " ".join(label(run.vocab[i]) for i in np.argsort(-s)[:k])


def swap(va, vb):
    """The paper's lens-coordinate swap: read 2 coordinates, exchange them, keep everything orthogonal to them."""
    a = np.stack([va, vb], 1)
    pinv = np.linalg.pinv(a)

    def fn(x):
        c = x @ pinv.T
        return x + (c[:, ::-1] - c) @ a.T
    return fn


def has_word(sentence, word):
    return re.search(rf"(^|\W){re.escape(word)}(\W|$)", sentence) is not None


def subject_facts(run, s):
    return {rel: o for rel, s2, o in run.cfg["facts"] if s2 == s}


def swap_rows(run, a, b, layer, va, vb, cache):
    """Ask every question about a, with a's and b's coordinates swapped at layer: does it answer with b's facts?"""
    fa, fb = subject_facts(run, a), subject_facts(run, b)
    rows = []
    for rel in fa:
        if rel in fb and fa[rel] != fb[rel]:
            q = corpus.QUESTIONS[rel][0].format(s=a, o=fa[rel], a=corpus.article(a))
            if q not in cache:
                cache[q] = text(run, answer(run, q)[1])
            before = cache[q]
            after = text(run, answer(run, q, edit=(layer, swap(va, vb)))[1])
            rows.append((rel, before, after, has_word(after, fb[rel]) and not has_word(after, fa[rel])))
    return rows


def cmd_jlens(run, q):
    mats = jlens_mats(run)
    ids, out = answer(run, q)
    rec = forward(run, ids + out)[1]
    print(f"{q!r} -> {text(run, out)!r}   (J-lens top 3 per layer; logit lens = the same with J = I)")
    last = len(mats) - 1
    print("  pos  token          " + "".join(f"L{l} J-lens{'':20s}" for l in range(len(mats))) + f"L{last} logit lens")
    eye = np.eye(run.m["d"])
    for t, tok in enumerate(ids + out):
        cells = [f"{lens_read(run, j, rec['resid'][l][t]):28s}" for l, j in enumerate(mats)]
        print(f"  {t:3d}  {label(run.vocab[tok]):14s} " + "".join(cells) + lens_read(run, eye, rec["resid"][last][t]))


def cmd_jswap(run, a, b, layer):
    j = jlens_mats(run)[layer]
    v = jvectors(run, j)
    rows = swap_rows(run, a, b, layer, v[run.word_id(a)], v[run.word_id(b)], {})
    if not rows:
        raise SystemExit(f"'{a}' and '{b}' share no relation with different answers")
    print(f"swap {a} ↔ {b} in J-lens coordinates at layer {layer} (all positions), then ask about {a}:")
    for rel, before, after, ok in rows:
        print(f"  {rel:10s} {before!r}\n  {'':10s} → {after!r}   {'✓ follows the swap' if ok else '✗'}")
    print(f"  {sum(r[3] for r in rows)}/{len(rows)} questions follow the swap (paper's France→China test: all 4)")


def pursuit(vn, h, k):
    """Greedy non-negative pursuit over unit vectors vn [N, d]: share of |h|² explained by k of them."""
    sel, r, c = [], h.copy(), np.zeros(0)
    for _ in range(k):
        s = vn @ r
        s[sel] = -np.inf
        best = int(s.argmax())
        if s[best] <= 0:
            break
        sel.append(best)
        c = np.maximum(np.linalg.lstsq(vn[sel].T, h, rcond=None)[0], 0)
        r = h - vn[sel].T @ c
    return 1 - float(r @ r) / float(h @ h + 1e-12)


def swap_rate(run, layer, vecs, pairs, cache):
    rows = [row for a, b in pairs for row in swap_rows(run, a, b, layer, vecs(a), vecs(b), cache)]
    return sum(r[3] for r in rows) / max(1, len(rows)), len(rows)


def cmd_jspace(run, n_pairs, k):
    """Prototype emergence check: causal (J swap ≫ random swap) and small (J part of the activation is low)."""
    rng = np.random.default_rng(0)
    subjects = {t: sorted({s for _, s, _ in run.cfg["facts"] if run.types.get(s) == t}) for t in ("country", "animal")}
    pairs = [tuple(rng.choice(subjects[t], 2, replace=False)) for t in subjects for _ in range(n_pairs // 2)]
    cache, mats = {}, jlens_mats(run)
    print(f"jspace  step {run.step} · {len(pairs)} subject pairs · pursuit k={k}  (prototype heuristics, see jspace.py LATER)")
    for l, j in enumerate(mats):
        v, eye_v = jvectors(run, j), jvectors(run, np.eye(run.m["d"]))
        rand = rng.standard_normal(v.shape) * np.linalg.norm(v, axis=1, keepdims=True) / np.sqrt(v.shape[1])
        rates = [swap_rate(run, l, f, pairs, cache)[0] for f in (lambda w: v[run.word_id(w)],
                 lambda w: eye_v[run.word_id(w)], lambda w: rand[run.word_id(w)])]
        hs = [forward(run, pr["ids"])[1]["resid"][l][-1] for pr in run.cfg["probes"][::9]]
        vn, rn = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) for x in (v, rand))
        frac_j, frac_r = (np.mean([pursuit(m, h, k) for h in hs]) for m in (vn, rn))
        causal, small = rates[0] >= 0.5 and rates[2] <= 0.1, frac_j <= 0.2
        verdict = ("input layer (raw word embedding): a swap here just swaps the word, not counted" if l == 0 else
                   "workspace-like ✓" if causal and small else
                   f"not yet: {'causal ✓' if causal else 'causal ✗'} · {'small ✓' if small else 'small ✗'}")
        print(f"  L{l}  swap J {rates[0]:.0%} · logit-lens {rates[1]:.0%} · random {rates[2]:.0%}   "
              f"J part of activation {frac_j:.0%} (random set {frac_r:.0%})   → {verdict}")


def emit(lines, budget):
    out, used = [], 0
    for line in lines:
        if used + len(line) > budget * 4:
            out.append(f"… ({len(lines) - len(out)} more lines; raise --budget)")
            break
        out.append(line)
        used += len(line)
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--step", type=int, help="look at an earlier checkpoint")
    ap.add_argument("--base", action="store_true", help="ignore hand-written edits (the model as trained)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("atlas").add_argument("--budget", type=int, default=400, help="max output, in tokens (approx)")
    sub.add_parser("ask").add_argument("question")
    tr = sub.add_parser("trace")
    tr.add_argument("question")
    tr.add_argument("--top", type=int, default=6)
    sub.add_parser("neuron").add_argument("unit")
    sub.add_parser("head").add_argument("unit")
    sub.add_parser("word").add_argument("word", nargs="+")
    df = sub.add_parser("diff")
    df.add_argument("a", type=int)
    df.add_argument("b", type=int)
    ab = sub.add_parser("ablate")
    ab.add_argument("unit")
    ab.add_argument("--ask", nargs="+", required=True)
    sub.add_parser("timeline")
    sub.add_parser("failures")
    ed = sub.add_parser("edit")
    ed.add_argument("fact", nargs="+", help="'peru capital lima' or a fact number from failures")
    ed.add_argument("--mode", choices=["protect", "rome", "sparse"], default="protect")
    ed.add_argument("--rows", type=int, default=8, help="sparse mode: how many neurons may change")
    ed.add_argument("--margin", type=float, default=2.0, help="how far the right word should lead the wrong one")
    ed.add_argument("--near", type=int, default=128, help="protect mode: how many similar keys of other facts stay fixed")
    ed.add_argument("--allow-damage", action="store_true", help="keep an edit even if it breaks other facts")
    sub.add_parser("check")
    sub.add_parser("edits")
    sub.add_parser("undo").add_argument("id", type=int, nargs="?")
    sub.add_parser("jlens").add_argument("question")
    js = sub.add_parser("jswap")
    js.add_argument("a")
    js.add_argument("b")
    js.add_argument("--layer", type=int, default=1)
    jp = sub.add_parser("jspace")
    jp.add_argument("--pairs", type=int, default=30)
    jp.add_argument("--k", type=int, default=5)
    pg = sub.add_parser("png")
    pg.add_argument("matrix", help="emb, unemb, L1.mlp.w1, L1.mlp.w2, L0.attn.q, ...")
    pg.add_argument("--out")
    args = ap.parse_args()
    run = Run(args.run, args.step, edits=not args.base)
    if args.cmd in ("failures", "edit", "check", "edits", "undo"):
        import wsedit  # editing lives in its own module
        return {"failures": lambda: wsedit.cmd_failures(run),
                "edit": lambda: wsedit.cmd_edit(run, " ".join(args.fact), args.mode, args.rows, args.margin,
                                                args.near, not args.allow_damage),
                "check": lambda: wsedit.cmd_check(run),
                "edits": lambda: wsedit.cmd_edits(run),
                "undo": lambda: wsedit.cmd_undo(run, args.id)}[args.cmd]()
    {"atlas": lambda: cmd_atlas(run, args.budget),
     "ask": lambda: print(text(run, answer(run, args.question)[1])),
     "trace": lambda: cmd_trace(run, args.question, args.top),
     "neuron": lambda: cmd_neuron(run, args.unit),
     "head": lambda: cmd_head(run, args.unit),
     "word": lambda: cmd_word(run, " ".join(args.word)),
     "diff": lambda: cmd_diff(run, args.a, args.b),
     "ablate": lambda: cmd_ablate(run, args.unit, args.ask),
     "png": lambda: cmd_png(run, args.matrix, args.out),
     "timeline": lambda: cmd_timeline(run),
     "jlens": lambda: cmd_jlens(run, args.question),
     "jswap": lambda: cmd_jswap(run, args.a, args.b, args.layer),
     "jspace": lambda: cmd_jspace(run, args.pairs, args.k)}[args.cmd]()


if __name__ == "__main__":
    main()
