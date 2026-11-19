"""wsv editing: fix what the model gets wrong by writing weights directly (closed form, no gradient descent).

  wsv failures                         facts the model gets wrong, and the first wrong word
  wsv edit "peru capital lima"         write a rank-1 change into the last MLP's output weights
          [--mode rome|sparse] [--rows 8] [--margin 4]
  wsv check                            every fact, edited model vs the base model: fixed / broken
  wsv edits | wsv undo [ID]            list / remove edits (stored in run/edits/, never mixed into checkpoints)

How an edit is written (key -> value, as in ROME, Meng et al. 2022):
  key   k = the last MLP layer's hidden activation where the wrong word is produced
  value δ = the residual-stream change that moves the logit from the wrong word to the right one
            (exact through the final LayerNorm to first order; repeated until the whole answer is right)
  protect (default) least-norm ΔW2 with  target key -> δ  and the `near` most similar keys of OTHER facts -> 0;
         then `check`: if anything broke, those facts' keys join the protected set and the edit is redone
  rome   ΔW2 = (C⁻¹k) δᵀ / (kᵀC⁻¹k), C = covariance of hidden activations: least disturbance on average
  sparse only the `rows` most active neurons change: fewest weights touched, most collateral damage
"""
import json
import os
import time

import numpy as np

import corpus
from wsv import EOT, answer, forward, label, prompt, text


# ---------- facts as questions and expected answers ----------
def question(run, f):
    rel, s, o = run.cfg["facts"][f]
    return corpus.QUESTIONS[rel][0].format(s=s, o=o, a=corpus.article(s))


def expected(run, f):
    rel, s, o = run.cfg["facts"][f]
    return [run.index[t] for t in corpus.answer_tokens(rel, s, o)]


def teacher_seq(run, f):
    """[bos] question [end] answer [end], and the first position whose next token is an answer word."""
    p = prompt(run, question(run, f))
    return p + expected(run, f) + [run.index[EOT]], len(p) - 1


def first_wrong(run, f):
    seq, start = teacher_seq(run, f)
    logits = forward(run, seq[:-1])[0]
    for p in range(start, len(seq) - 1):
        got = int(logits[p].argmax())
        if got != seq[p + 1]:
            return p, seq[p + 1], got, float(logits[p, seq[p + 1]] - logits[p, got])
    return None


def evaluate(run):
    """Greedy answer to every fact's trained question: [(fact, right?, what it said)]."""
    return [(f, out == expected(run, f), text(run, out))
            for f in range(len(run.cfg["facts"])) for out in [answer(run, question(run, f))[1]]]


def base_results(run):
    cache = os.path.join(run.path, f"eval_base@{run.step}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    saved = {name: arr for name, arr in run.p.items()}
    run.p = run.params(run.step)                         # the model without any edits
    res = [[f, ok, said] for f, ok, said in evaluate(run)]
    run.p = saved
    json.dump(res, open(cache, "w"))
    return res


def find_fact(run, spec):
    if spec.isdigit():
        return int(spec)
    for i, (rel, s, o) in enumerate(run.cfg["facts"]):
        if f"{s} {rel} {o}".startswith(spec):
            return i
    raise SystemExit(f"no fact matches '{spec}' (use 'subject relation object', e.g. 'peru capital lima')")


def fact_name(run, f):
    rel, s, o = run.cfg["facts"][f]
    return f"{s} {rel} {o}"


# ---------- writing an edit ----------
def key_cov(run, l):
    """Covariance of the layer-l MLP hidden activations over every answer position of every fact."""
    keys = []
    for f in range(len(run.cfg["facts"])):
        seq, start = teacher_seq(run, f)
        keys.append(forward(run, seq[:-1])[1]["layers"][l]["hid"][start:])
    k = np.concatenate(keys)
    c = k.T @ k / len(k)
    return c + 1e-2 * np.trace(c) / len(c) * np.eye(len(c))


def answer_keys(run, l, skip):
    """Hidden activations at every answer position of every other fact, and which fact each belongs to."""
    keys, owners = [], []
    for f in range(len(run.cfg["facts"])):
        if f != skip:
            seq, start = teacher_seq(run, f)
            hid = forward(run, seq[:-1])[1]["layers"][l]["hid"][start:]
            keys.append(hid)
            owners += [f] * len(hid)
    return np.concatenate(keys).astype(float), np.array(owners)


def protected_solve(k, delta, keys, owners, near, extra, lam=1e-3):
    """Least-norm ΔW with k -> delta and every protected key -> 0 (protected = most similar keys + `extra` facts)."""
    sims = keys @ k / (np.linalg.norm(keys, axis=1) * np.linalg.norm(k) + 1e-9)
    pick = set(np.argsort(-sims)[:near]) | set(np.flatnonzero(np.isin(owners, list(extra))))
    a = np.vstack([k[None], keys[sorted(pick)]])
    t = np.zeros((len(a), len(delta)))
    t[0] = delta
    return a.T @ np.linalg.solve(a @ a.T + lam * np.eye(len(a)), t)


def value_vector(run, rec, logits, p, want, got, margin):
    """Residual change at position p that makes logit(want) − logit(got) = margin (first order, through LN_final)."""
    x = rec["x"][p]
    a = run.p["lnf.weight"] * (run.p["wu"][:, want] - run.p["wu"][:, got])
    a = a - a.mean()
    gap = logits[p, want] - logits[p, got]
    return a * (margin - gap) * np.sqrt(x.var() + 1e-5) / (a @ a)


def rank1(k, delta, mode, rows, cinv):
    if mode == "rome":
        u = cinv @ k
        return np.outer(u, delta) / (k @ u)
    top = [i for i in np.argsort(-k)[:rows] if k[i] > 0]
    d = np.zeros((len(k), len(delta)))
    d[top] = np.outer(k[top], delta) / (k[top] @ k[top])
    return d


def write_edit(run, f, mode, rows, margin, near=64, extra=(), iters=6):
    """Edit the last MLP's output weights until the fact's whole answer comes out right (or iters run out)."""
    l = run.m["layers"] - 1
    name = f"blocks.{l}.w2"
    cinv = np.linalg.inv(key_cov(run, l)) if mode == "rome" else None
    keys, owners = answer_keys(run, l, f) if mode == "protect" else (None, None)
    seq, start = teacher_seq(run, f)
    total, steps = np.zeros_like(run.p[name], dtype=float), []
    for _ in range(iters):
        logits, rec = forward(run, seq[:-1])
        wrong = [p for p in range(start, len(seq) - 1) if int(logits[p].argmax()) != seq[p + 1]]
        if not wrong:
            break
        p = wrong[0]
        want, got = seq[p + 1], int(logits[p].argmax())
        k = rec["layers"][l]["hid"][p].astype(float)
        if not k.any():
            break
        delta = value_vector(run, rec, logits, p, want, got, margin)
        d = (protected_solve(k, delta, keys, owners, near, extra) if mode == "protect"
             else rank1(k, delta, mode, rows, cinv))
        run.p = {**run.p, name: run.p[name] + d.astype(run.p[name].dtype)}
        total += d
        steps.append(f"pos {p}: '{label(run.vocab[got])}' → '{label(run.vocab[want])}'")
    return name, total, steps, first_wrong(run, f) is None


# ---------- storage ----------
def edits_dir(run):
    return os.path.join(run.path, "edits")


def load_index(run):
    path = os.path.join(edits_dir(run), "index.json")
    return json.load(open(path)) if os.path.exists(path) else []


def save_index(run, index):
    os.makedirs(edits_dir(run), exist_ok=True)
    json.dump(index, open(os.path.join(edits_dir(run), "index.json"), "w"), indent=1)


def n_params(run):
    return sum(v.size for v in run.p.values())


# ---------- commands ----------
def cmd_failures(run, limit=40):
    wrong = [(f, said) for f, ok, said in evaluate(run) if not ok]
    print(f"failures  {len(wrong)}/{len(run.cfg['facts'])} facts wrong  (trained question, greedy answer"
          f"{', ' + str(len(run.edits)) + ' edits applied' if run.edits else ''})")
    for f, said in wrong[:limit]:
        fw = first_wrong(run, f)
        why = (f"'{label(run.vocab[fw[2]])}' instead of '{label(run.vocab[fw[1]])}' (gap {fw[3]:+.1f})" if fw else "")
        print(f"  {f:3d}  {fact_name(run, f):34s} says {said!r:44s} {why}")


def cmd_edit(run, spec, mode, rows, margin, near=128, strict=True):
    if run.step != run.steps[-1]:
        raise SystemExit("edits apply to the final checkpoint only")
    f = find_fact(run, spec)
    before = text(run, answer(run, question(run, f))[1])
    right_before = {g for g, ok, _ in evaluate(run) if ok}
    start_p, extra, rounds, best = run.p, set(), [], None
    for _ in range(4 if mode == "protect" else 1):  # edit, check, protect what broke, redo; keep the best round
        run.p = start_p
        name, total, steps, fixed = write_edit(run, f, mode, rows, margin, near=near, extra=extra)
        broken = right_before - {g for g, ok, _ in evaluate(run) if ok}
        rounds.append(len(broken))
        if best is None or (fixed, -len(broken)) > (best[3], -len(best[4])):
            best = (run.p, total, steps, fixed, broken)
        if not broken or not steps:
            break
        extra |= broken
    run.p, total, steps, fixed, broken = best
    if strict and (broken or not fixed) and steps:
        run.p = start_p
        return print(f"edit {fact_name(run, f)}: REJECTED, the best version {'fixes it but ' if fixed else ''}"
                     f"breaks {len(broken)} other facts (rounds: {' → '.join(map(str, rounds))}); nothing written")
    if not steps:
        return print(f"{fact_name(run, f)}: already right ({before!r}), nothing to write")
    index = load_index(run)
    eid = max([e["id"] for e in index], default=0) + 1
    os.makedirs(edits_dir(run), exist_ok=True)
    np.savez(os.path.join(edits_dir(run), f"{eid}.npz"), **{name: total.astype(np.float32)})
    touched = int(np.count_nonzero(total))
    index.append({"id": eid, "fact": fact_name(run, f), "mode": mode, "rows": rows, "margin": margin, "param": name,
                  "weights": touched, "fixed": fixed, "steps": steps, "created": time.strftime("%Y-%m-%d %H:%M")})
    save_index(run, index)
    after = text(run, answer(run, question(run, f))[1])
    print(f"edit #{eid}  {fact_name(run, f)}  ({mode}{f', {rows} rows' if mode == 'sparse' else ''})")
    print(f"  before {before!r}\n  after  {after!r}   {'✓ fixed' if fixed else '✗ not fully fixed'}")
    print(f"  wrote {name}: {touched:,} weights ({100 * touched / n_params(run):.2f}% of the model) · " + " · ".join(steps))
    print(f"  other facts broken per round: {' → '.join(map(str, rounds))}"
          + ("" if mode == "protect" else "   (run `wsv check` for the full picture)"))


def cmd_check(run):
    base = {f: (ok, said) for f, ok, said in base_results(run)}
    now = evaluate(run)
    fixed = [f for f, ok, _ in now if ok and not base[f][0]]
    broken = [(f, said) for f, ok, said in now if not ok and base[f][0]]
    masks = {}
    for e in run.edits:  # distinct weights touched across all edits
        with np.load(os.path.join(edits_dir(run), f"{e['id']}.npz")) as z:
            for k in z.files:
                masks[k] = masks.get(k, np.zeros(z[k].shape, bool)) | (z[k] != 0)
    total_w = int(sum(m.sum() for m in masks.values()))
    n = len(now)
    print(f"check  {len(run.edits)} edits · {total_w:,} weights written ({100 * total_w / n_params(run):.2f}%) · vs the base model")
    print(f"  facts right  {sum(ok for ok, _ in base.values())} → {sum(ok for _, ok, _ in now)} / {n}"
          f"   (+{len(fixed)} fixed, −{len(broken)} broken)")
    if fixed:
        print("  fixed   " + " · ".join(fact_name(run, f) for f in fixed))
    for f, said in broken:
        print(f"  broken  {fact_name(run, f):34s} was {base[f][1]!r} → now {said!r}")


def cmd_edits(run):
    index = load_index(run)
    if not index:
        return print("no edits")
    for e in index:
        print(f"  #{e['id']:<3d} {e['fact']:34s} {e['mode']:6s} {e['weights']:6,d} weights  "
              f"{'fixed' if e['fixed'] else 'partial'}  {e['created']}")


def cmd_undo(run, eid):
    index = load_index(run)
    if not index:
        return print("no edits to undo")
    eid = eid if eid is not None else index[-1]["id"]
    keep = [e for e in index if e["id"] != eid]
    if len(keep) == len(index):
        raise SystemExit(f"no edit #{eid}")
    os.remove(os.path.join(edits_dir(run), f"{eid}.npz"))
    save_index(run, keep)
    print(f"removed edit #{eid}; {len(keep)} edits left")
