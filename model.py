"""hello-llm inference: tokenizer + plain GPT forward pass (embed -> [attention + MLP] x L -> unembed).

This file knows nothing about countries, animals or answers. Every behavior lives in
weights.npz, which build.py writes by hand: no training, no data, no gradients.
Run: python model.py
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights.npz")
MAX_USER, MAX_REPLY, MAX_PHRASE = 60, 250, 3
BOS, EOT, XSTART = "^", "\n", " "


def load(path=WEIGHTS):
    with np.load(path) as z:
        w = {k: z[k] for k in z.files}
    w["vocab"] = [str(t) for t in w["vocab"]]
    w["index"] = {t: i for i, t in enumerate(w["vocab"])}
    w["fast"] = [compact(w, l) for l in range(int(w["n_layer"]))]
    w["wu_rows"] = nonzero_rows(w["wu"])
    return w


def nonzero_rows(a):
    return np.flatnonzero(np.any(a != 0, axis=1))


def compact(w, l):
    """Same math, skipping residual dims a layer never reads or writes (the weights are ~99.9% zeros)."""
    wq, wk, wv, wo = (w[f"{k}{l}"] for k in ("wq", "wk", "wv", "wo"))
    attn = None
    if len(wq):
        rows = np.flatnonzero(np.any((wq != 0) | (wk != 0) | (wv != 0), axis=(0, 2)))
        cols = np.flatnonzero(np.any(wo != 0, axis=(0, 1)))
        attn = (rows, wq[:, rows], wk[:, rows], wv[:, rows], wo[:, :, cols], cols, np.sqrt(wq.shape[-1]))
    w1, w2 = w[f"w1{l}"], w[f"w2{l}"]
    rows, cols = nonzero_rows(w1), nonzero_rows(w2.T)
    return attn, (rows, w1[rows], w[f"b1{l}"], w2[:, cols], cols)


def tokenize(w, text):
    """Longest known phrase (up to 3 words) -> one token; unknown words are spelled out char by char."""
    idx, out = w["index"], []
    pieces = re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())
    i = 0
    while i < len(pieces):
        if not pieces[i].isalnum():
            out += [idx[pieces[i]]] if pieces[i] in idx else []
            i += 1
            continue
        for n in range(MAX_PHRASE, 0, -1):
            span = pieces[i:i + n]
            phrase = " " + " ".join(span)
            if len(span) == n and all(p.isalnum() for p in span) and phrase in idx:
                out.append(idx[phrase])
                i += n
                break
        else:
            out += [idx[XSTART]] + [idx[c] for c in pieces[i]]
            i += 1
    return out[:MAX_USER]


def step(w, tok, pos, cache):
    """Run one token through the network (KV-cached); return logits for the next token."""
    x = w["wte"][tok] + w["wpe"][pos]
    for (attn, (rows, w1, b1, w2, cols)), (ks, vs) in zip(w["fast"], cache):
        if attn:
            ar, wq, wk, wv, wo, ac, scale = attn
            xr = x[ar]
            q = xr @ wq                                         # (heads, d_head)
            ks.append(xr @ wk)
            vs.append(xr @ wv)
            k, v = np.stack(ks, 1), np.stack(vs, 1)             # (heads, t, d_head)
            s = np.einsum("he,hte->ht", q, k) / scale
            a = np.exp(s - s.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            x[ac] += np.einsum("he,hed->d", np.einsum("ht,hte->he", a, v), wo)
        x[cols] += np.maximum(0, x[rows] @ w1 + b1) @ w2
    return x[w["wu_rows"]] @ w["wu"][w["wu_rows"]]


def generate(w, text):
    """Yield the reply to one user message, token by token (greedy decoding)."""
    toks = [w["index"][BOS]] + tokenize(w, text) + [w["index"][EOT]]
    cache = [([], []) for _ in range(int(w["n_layer"]))]
    for pos, tok in enumerate(toks):
        logits = step(w, tok, pos, cache)
    for pos in range(len(toks), len(toks) + MAX_REPLY):
        tok = int(np.argmax(logits))
        if w["vocab"][tok] == EOT:
            return
        yield w["vocab"][tok]
        logits = step(w, tok, pos, cache)


def reply(w, text):
    return "".join(generate(w, text)).strip()


def summary(w):
    n_layer = int(w["n_layer"])
    params = [v for k, v in w.items() if isinstance(v, np.ndarray) and v.dtype == np.float32]
    total, nonzero = sum(p.size for p in params), sum(int(np.count_nonzero(p)) for p in params)
    heads = sum(len(w[f"wq{l}"]) for l in range(n_layer))
    neurons = sum(w[f"b1{l}"].size for l in range(n_layer))
    return (f"{n_layer} layers, {heads} heads, {neurons:,} neurons, d_model {w['wte'].shape[1]}, "
            f"vocab {len(w['vocab'])} | {total / 1e6:.1f}M params, {nonzero:,} nonzero, all hand-written")


def main():
    try:
        w = load()
    except FileNotFoundError:
        raise SystemExit("no weights.npz yet: run python build.py first")
    print(f"hello-llm | {summary(w)} | ctrl-d quits")
    while True:
        try:
            msg = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print("bot>", end="", flush=True)
        for tok in generate(w, msg):
            print(tok, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
