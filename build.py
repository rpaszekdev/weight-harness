"""Compile hello-llm v2: facts + lexicon -> hand-written transformer weights. No training, no gradients.

Weight-space programming (rikiyat.github.io/blog/weight-space-programming/). The weights
hold atoms only: 315 facts (one MLP neuron each), word features, and 7 relation frames.
No answer sentence is stored; every reply is assembled while it is generated:

  L0   attn: find the end of the user turn            mlp: answer-position counter
  L1   attn: slot heads grab the first/last entity of each type, bag of cue words,
             yes/no opener, first unknown word          mlp: clamp counts -> 0/1
  L2-3 mlp : pick one relation (argmax over votes from cue words and entity types)
  L4   mlp : HOP. a city asked about by continent/language -> its country ("is ottawa in europe?")
  L5   mlp : FACTS. neuron(subject AND relation) writes subject + object; read backwards
             when only the object was mentioned
  L6-8 mlp : yes/no check, found anything?, a/an, answer mode
  L9   attn: pointer that copies an unknown word       mlp: frame grammar (mode, position) -> word/slot
  L10  mlp : fill slots: subject/object -> its word, a/an, copied chars
LISTS ("countries in europe", "capitals u know", "cities in europe"): L4 picks the filter, L5 marks
  every matching entity, L6 optionally takes one more step (each country's capital/language/...),
  an L1 head sees what the answer has said so far, L7-8 keep the first one not yet said,
  L10 says it, then "," while any are left, else ".".
Run: python build.py
"""
import json
import os

import numpy as np

import lexicon as lx

HERE = os.path.dirname(os.path.abspath(__file__))
N_LAYER, D_HEAD = 11, 128
L_HOP, L_FACT, L_CMP, L_FLAG, L_MODE, L_FRAME, L_FILL = 4, 5, 6, 7, 8, 9, 10
T_MAX, K_MAX = 384, 40                  # max context tokens, longest fixed answer part
BETA_Q, BETA_T, GAMMA = 50.0, 5000.0, 20.0  # pointer sharpness, type match, first/last tie-break
BIG, BIG_K, GAIN, GAIN_A = 1e5, 1000.0, 128.0, 1024.0  # masks, counter gate, bag gains (question, answer)
S = np.sqrt(D_HEAD)                     # cancels the 1/sqrt(d_head) in attention
BOS, EOT, XSTART = "^", "\n", " "       # begin, end of turn, start of a spelled-out word


def load_facts():
    doc = json.load(open(os.path.join(HERE, "facts.json")))
    ents, facts = {t: [] for t in lx.TYPES}, []

    def ent(t, name):
        if name not in ents[t]:
            ents[t].append(name)
        return t, name

    for c in doc["countries"]:
        s = ent("country", c["country"])
        facts += [("capital", s, ent("city", c["capital"])), ("continent", s, ent("continent", c["continent"])),
                  ("language", s, ent("language", c["language"]))]
        if c.get("currency"):
            facts.append(("currency", s, ent("currency", c["currency"])))
    for a in doc["animals"]:
        s = ent("animal", a["animal"])
        if a["sound"]:
            facts.append(("sound", s, ent("sound", a["sound"])))
        facts += [("legs", s, ent("number", str(a["legs"]))), ("class", s, ent("class", a["class"])),
                  ("habitat", s, ent("habitat", a["habitat"])), ("diet", s, ent("diet", a["diet"]))]
    for x, y in doc["opposites"]:
        facts += [("opposite", ent("adj", x), ent("adj", y)), ("opposite", ent("adj", y), ent("adj", x))]
    return ents, facts


ENTS, FACTS = load_facts()
ENT_LIST = [(t, n) for t in lx.TYPES for n in ENTS[t]]

# ---------- residual stream layout ----------
NAMES = []


def alloc(name):
    NAMES.append(name)
    return len(NAMES) - 1


def allocs(prefix, keys):
    return {k: alloc(f"{prefix}:{k}") for k in keys}


ONE, POS, POS2 = alloc("one"), alloc("pos"), alloc("pos^2")
ENT = allocs("ent", ENT_LIST)                       # which entity this token names
TYPE = allocs("type", lx.TYPES)
CUE = allocs("cue", lx.CUES)
GREET, YNW, XS, CONT, DOT, SEP = (alloc(n) for n in ("greet", "yn-word", "spell-start", "char", "dot", "sep"))
LEAD = alloc("lead")                                # can open the sentence (not a greeting or punctuation)
ISENT, COMMA, BANG = alloc("is-entity"), alloc("comma"), alloc("bang")
TN = allocs("list-noun", lx.LIST_TYPES)             # "countries", "capitals", ... asks for a whole list
CH = allocs("ch", lx.CHARS)
AFTER, SEPPOS = alloc("after-sep"), alloc("sep-pos")
OFF = [alloc(f"off:{k}") for k in range(K_MAX + 1)]
FIRST = {t: allocs(f"first-{t}", ENTS[t]) for t in lx.SUBJ_TYPES}
LAST = {t: allocs(f"last-{t}", ENTS[t]) for t in lx.OBJ_TYPES}
BAG = [CUE[c] for c in lx.CUES] + [TYPE[t] for t in lx.TYPES] + [GREET] + list(TN.values())
WSUM = [alloc(f"wsum:{NAMES[d]}") for d in BAG]
HAS = dict(zip(BAG, (alloc(f"has:{NAMES[d]}") for d in BAG)))  # keyed by the feature it summarizes
ADJ_SUM, ADJ_HAS = allocs("adjsum", ENTS["adj"]), allocs("adjhas", ENTS["adj"])  # which adjectives occur
YNQ, XPOS, XFLAG = alloc("yn-question"), alloc("unknown-pos"), alloc("unknown-word")
INV = {to: allocs(f"inv-{to}", ENTS[to]) for ts, to, back in lx.RELATIONS.values() if back and ts != to}
GT = allocs("gt", [(r, r2) for r in lx.PRIORITY for r2 in lx.PRIORITY if r != r2])
REL = allocs("rel", lx.PRIORITY)
SUBJ = allocs("subj", [e for e in ENT_LIST if e[0] in lx.SUBJ_TYPES])
OBJ = allocs("obj", [e for e in ENT_LIST if e[0] in lx.OBJ_TYPES])
OBJM, SUBJM, TWO_ADJ, MATCH, FOUNDR, VOWS, VOWO, YNA, FOUND, YES, NO, MX, MG, MI = (alloc(n) for n in (
    "obj-mentioned", "subj-mentioned", "two-adjectives", "match", "found-raw", "vowel-s", "vowel-o",
    "yn-active", "found", "yes", "no", "mode-unknown", "mode-greet", "mode-idk"))
SLOT = allocs("slot", ["S", "O", "AS", "AO"])
PH1, PHR = alloc("copy-first"), alloc("copy-rest")
COPY = allocs("copy", lx.CHARS)
COPYCONT = alloc("copy-is-char")
LIST_ENTS = [e for e in ENT_LIST if e[0] in lx.LIST_TYPES]
SAID, CAND, REMAIN, PICK = (allocs(n, LIST_ENTS) for n in ("said", "cand", "remain", "pick"))
TL = allocs("list-type", lx.LIST_TYPES)             # "which countries do you know?" -> list all of a type
LOOPSEP, LISTREL, LISTANY, ANYREM, ML = (alloc(n) for n in (
    "list-sep", "list-by-relation", "list-any", "any-left", "mode-list"))
SAIDC = allocs("said-clamped", LIST_ENTS)
TNANY = alloc("list-noun-any")
LISTF = allocs("list-filter", [r for r, (_, _, back) in lx.RELATIONS.items() if not back])
ATTR_RELS = {r: to for r, (ts, to, _) in lx.RELATIONS.items() if ts == "country" and to in lx.LIST_TYPES}


def words(text):
    """Frame text -> tokens: words carry a leading space, punctuation does not."""
    return [w if w in lx.PUNCT else " " + w for w in text.split()]


def vocab_features():
    feats = {}

    def put(tok, *dims):
        feats.setdefault(tok, {ONE: 1}).update({d: 1 for d in dims})

    put(BOS)
    put(EOT, SEP)
    put(XSTART, XS)
    for c in lx.CHARS:
        put(c, CONT, CH[c])
    for p in lx.PUNCT:
        put(p)
    feats["."][DOT], feats[","][COMMA], feats["!"][BANG] = 1, 1, 1
    for w in lx.FUNCTION_WORDS.split():
        put(" " + w)
    for w in lx.YN_WORDS.split():
        put(" " + w, YNW)
    for w in lx.GREET_WORDS.split() + lx.GREET_PHRASES:
        put(" " + w, GREET)
    for c, (_, _, ws) in lx.CUES.items():
        for w in ws.split():
            put(" " + w, CUE[c])
    for t, n in ENT_LIST:
        names = [n] + lx.SYNONYMS.get((t, n), [])
        surfaces = names + ([lx.plural(x) for x in names if " " not in x] if t in lx.PLURAL_TYPES else []) \
            + (lx.verb_forms(n) if t == "sound" else [])
        for w in surfaces:
            other = [d for d in feats.get(" " + w, {}) if NAMES[d].startswith("ent:") and d != ENT[(t, n)]]
            assert not other, f"'{w}' would name 2 entities"
            put(" " + w, ENT[(t, n)], TYPE[t], ISENT)
    for t, ws in lx.LIST_TYPES.items():
        for w in ws.split() + [f"{q} {n}" for n in lx.LIST_SINGULAR[t].split() for q in ("every", "each")]:
            put(" " + w, TN[t])
    for text in list(lx.FRAMES.values()) + [lx.GREET_PREFIX, lx.YES_PREFIX, lx.NO_PREFIX, lx.UNKNOWN_LEAD,
                                            lx.IDK, "a an"]:
        for tok in words(text):
            if tok.strip() not in ("S", "O", "AS", "AO"):
                put(tok)
    for tok, f in feats.items():
        if tok not in (BOS, EOT) and tok not in lx.PUNCT and GREET not in f:
            f[LEAD] = 1
    return feats


FEATS = vocab_features()
VOCAB = list(FEATS)
NEXT = {tok: alloc(f"next:{tok!r}") for tok in VOCAB}  # next-token votes, read by the unembedding
D = len(NAMES)

wte, wpe, wu = np.zeros((len(VOCAB), D)), np.zeros((T_MAX, D)), np.zeros((D, len(VOCAB)))
attn = [[] for _ in range(N_LAYER)]  # per layer: list of heads (wq, wk, wv, wo)
mlp = [[] for _ in range(N_LAYER)]   # per layer: list of neurons (inputs {dim: w}, bias, outputs {dim: w})


def head(l):
    h = (np.zeros((D, D_HEAD)), np.zeros((D, D_HEAD)), np.zeros((D, D_HEAD)), np.zeros((D_HEAD, D)))
    attn[l].append(h)
    return h


def neuron(l, ins, bias, outs):
    mlp[l].append((ins, bias, outs))


def move(h, srcs, dsts, scale=1.0):
    """The head carries residual dims srcs into dims dsts, weighted by its attention."""
    srcs, dsts = list(srcs), list(dsts)
    assert len(srcs) == len(dsts) <= D_HEAD
    for i, (s, d) in enumerate(zip(srcs, dsts)):
        h[2][s, i], h[3][i, d] = scale, 1


def score_keys(h, key):
    """Constant query, so each key scores sum(coef * key[dim])."""
    h[0][ONE, 0] = S
    for d, c in key.items():
        h[1][d, 0] = c


def attend_to(h, target):
    """Pointer: attend to position sum(coef * x[dim]); score = -BETA_Q * (target - i)^2 + const."""
    for d, c in target.items():
        h[0][d, 0] = S * 2 * BETA_Q * c
    h[0][ONE, 1] = -S * BETA_Q
    h[1][POS, 0], h[1][POS2, 1] = 1, 1


def plus(*terms):
    out = {}
    for t in terms:
        for d, c in t.items():
            out[d] = out.get(d, 0) + c
    return out


def embeddings():
    for v, tok in enumerate(VOCAB):
        for d, c in FEATS[tok].items():
            wte[v, d] = c
        wu[NEXT[tok], v] = 30.0
    t = np.arange(T_MAX)
    wpe[:, POS], wpe[:, POS2] = t, t ** 2


def layer0():
    h = head(0)  # where is the end of the user turn, and are we past it?
    score_keys(h, {SEP: BETA_T})
    move(h, [SEP, POS], [AFTER, SEPPOS])
    k = {POS: 1, SEPPOS: -1, AFTER: BIG_K}  # answer position t - sep, far negative inside the question
    for j in range(K_MAX + 1):  # one-hot bump: relu(k-j+1) - 2 relu(k-j) + relu(k-j-1)
        for shift, out in ((1, 1), (0, -2), (-1, 1)):
            neuron(0, k, shift - j - BIG_K, {OFF[j]: out})


def layer1():
    h = head(1)  # bag of words: uniform over the question, scaled so any hit reaches >= 1
    score_keys(h, {AFTER: -BIG})
    move(h, BAG, WSUM, GAIN)
    for tie, table in ((-GAMMA, FIRST), (GAMMA, LAST)):  # slot heads: first / last entity of type t
        for t, slots in table.items():
            h = head(1)
            score_keys(h, {TYPE[t]: BETA_T, POS: tie, AFTER: -BIG})
            move(h, [ENT[(t, n)] for n in ENTS[t]], slots.values())
    h = head(1)  # does the question open (after any greeting) with is/are/do/would...?
    score_keys(h, {LEAD: BETA_T, POS: -GAMMA, AFTER: -BIG})
    move(h, [YNW], [YNQ])
    h = head(1)  # first spelled-out (unknown) word
    score_keys(h, {XS: BETA_T, POS: -GAMMA, AFTER: -BIG})
    move(h, [POS, XS], [XPOS, XFLAG])
    h = head(1)  # bag of adjectives (to check "is hot the opposite of cold?")
    score_keys(h, {AFTER: -BIG})
    move(h, [ENT[("adj", n)] for n in ENTS["adj"]], ADJ_SUM.values(), GAIN)
    pairs = list(zip(WSUM, (HAS[d] for d in BAG))) + [(ADJ_SUM[n], ADJ_HAS[n]) for n in ENTS["adj"]]
    for s, d in pairs:  # clamp(x, 0, 1) = relu(x) - relu(x - 1)
        neuron(1, {s: 1}, 0, {d: 1})
        neuron(1, {s: 1}, -1, {d: -1})


def votes():
    v = {r: {} for r in lx.PRIORITY}
    for c, (r, n, _) in lx.CUES.items():
        v[r] = plus(v[r], {HAS[CUE[c]]: n})
    for t, (r, n) in lx.TYPE_VOTES.items():
        v[r] = plus(v[r], {HAS[TYPE[t]]: n})
    return v


def layers2_3():
    v = votes()
    for (r, r2), d in GT.items():  # [votes(r) beat votes(r2)], ties go to the earlier relation
        diff = plus(v[r], {k: -c for k, c in v[r2].items()})
        tie = 0.5 if lx.PRIORITY.index(r) < lx.PRIORITY.index(r2) else -0.5
        neuron(2, diff, tie + 0.5, {d: 1})
        neuron(2, diff, tie - 0.5, {d: -1})
    for ts, to, back in lx.RELATIONS.values():  # object mentioned and no subject -> read the fact backwards
        if back and ts != to:
            for n, d in INV[to].items():
                neuron(2, {LAST[to][n]: 1, HAS[TYPE[ts]]: -1}, 0, {d: 1})
    for n in ENTS["adj"]:  # a second, different adjective (the object of "is hot the opposite of cold?")
        neuron(2, {LAST["adj"][n]: 1, FIRST["adj"][n]: -1}, 0, {TWO_ADJ: 1})
    for r in lx.PRIORITY:  # argmax = beats every other relation
        neuron(3, {GT[(r, r2)]: 1 for r2 in lx.PRIORITY if r2 != r}, 2 - len(lx.PRIORITY), {REL[r]: 1})


def hop():
    """City mentioned, no country, asked for continent/language: write its country into the subject slot."""
    for rel, s, o in FACTS:
        if rel == "capital":
            ins = {LAST["city"][o[1]]: 1, HAS[TYPE["country"]]: -1, REL["continent"]: 1, REL["language"]: 1}
            neuron(L_HOP, ins, -1, {FIRST["country"][s[1]]: 1})


def facts():
    for rel, s, o in FACTS:  # one neuron per fact: (subject or backwards-object) AND relation -> exactly 1
        ts, to, _ = lx.RELATIONS[rel]
        key = {FIRST[ts][s[1]]: 1, REL[rel]: 2}
        if to in INV:
            key[INV[to][o[1]]] = 1
        neuron(L_FACT, key, -2, {SUBJ[s]: 1, OBJ[o]: 1})
    for rel, (ts, to, _) in lx.RELATIONS.items():
        obj_seen = TWO_ADJ if ts == to else HAS[TYPE[to]]
        neuron(L_FACT, {REL[rel]: 1, obj_seen: 1}, -1, {OBJM: 1})
        neuron(L_FACT, {REL[rel]: 1, HAS[TYPE[ts]]: 1}, -1, {SUBJM: 1})


def checks():
    for e, d in OBJ.items():  # the object the fact gives == the one the question mentioned?
        seen = ADJ_HAS[e[1]] if e[0] == "adj" else LAST[e[0]][e[1]]  # adjectives: mentioned anywhere
        neuron(L_CMP, {d: 1, seen: 1}, -1, {MATCH: 1})
    neuron(L_CMP, dict.fromkeys(SUBJ.values(), 1), 0, {FOUNDR: 1})
    neuron(L_CMP, {d: 1 for e, d in SUBJ.items() if e[1][0] in "aeiou"}, 0, {VOWS: 1})
    neuron(L_CMP, {d: 1 for e, d in OBJ.items() if e[1][0] in "aeiou"}, 0, {VOWO: 1})
    # yes/no = an object to check was named AND (is/are/do... opener OR the subject was named too)
    yn = {OBJM: 2, YNQ: 1, SUBJM: 1}
    neuron(L_CMP, yn, -2, {YNA: 1})
    neuron(L_CMP, yn, -3, {YNA: -1})
    neuron(L_FLAG, {FOUNDR: 1}, 0, {FOUND: 1})
    neuron(L_FLAG, {FOUNDR: 1}, -1, {FOUND: -1})
    neuron(L_FLAG, {YNA: 2, MATCH: 1}, -2, {YES: 1})
    neuron(L_FLAG, {YNA: 2, MATCH: 1}, -3, {YES: -1})
    neuron(L_FLAG, {YNA: 1, MATCH: -1}, 0, {NO: 1})
    g, none = HAS[GREET], REL["none"]
    q = {FOUND: -1, LISTANY: -1}  # these modes only when no fact and no list answers
    neuron(L_MODE, plus(q, {XFLAG: 1, none: -1}), 0, {MX: 1})        # unknown word in a real question
    neuron(L_MODE, plus(q, {g: 1, XFLAG: -1}), 0, {MG: 1})           # just a greeting
    neuron(L_MODE, plus(q, {g: 1, none: 1, XFLAG: 1}), -2, {MG: 1})
    neuron(L_MODE, plus(q, {g: -1, XFLAG: -1}), 1, {MI: 1})          # nothing to go on
    neuron(L_MODE, plus(q, {none: 1, XFLAG: 1, g: -1}), -1, {MI: 1})


def token_out(tok):
    return {SLOT[tok.strip()]: 1} if tok.strip() in SLOT else {NEXT[tok]: 1}


def sequence(lits, outs):
    """Neuron k fires when every literal holds and the answer is at position k."""
    terms, const = plus(*(t for t, _ in lits)), sum(c for _, c in lits)
    for k, out in enumerate(outs):
        neuron(L_FRAME, plus(terms, {OFF[k]: 1}), const - len(lits), out)


def frames():
    g = HAS[GREET]
    greet = [({g: 1}, 0), ({g: -1}, 1)]  # literal "greeted", "not greeted"
    yn = {"": ({YES: -1, NO: -1}, 1), lx.YES_PREFIX: ({YES: 1}, 0), lx.NO_PREFIX: ({NO: 1}, 0)}
    for rel, frame in lx.FRAMES.items():
        for gi in (1, 0):
            for prefix, lit in yn.items():
                toks = words(lx.GREET_PREFIX) * gi + words(prefix) + words(frame) + [EOT]
                sequence([({FOUND: 1}, 0), ({REL[rel]: 1}, 0), greet[1 - gi], lit], [token_out(t) for t in toks])
    lead_len = len(words(lx.UNKNOWN_LEAD))
    for gi in (1, 0):
        lead = words(lx.GREET_PREFIX) * gi + words(lx.UNKNOWN_LEAD)
        outs = [token_out(t) for t in lead] + [{PH1: 1}] + [{PHR: 1}] * (K_MAX - len(lead))
        sequence([({MX: 1}, 0), greet[1 - gi]], outs)
    sequence([({MG: 1}, 0)], [token_out(t) for t in words(lx.GREET_PREFIX) + [EOT]])
    sequence([({MI: 1}, 0)], [token_out(t) for t in words(lx.IDK) + [EOT]])
    h = head(L_FRAME)  # pointer into the unknown word: answer position k -> word start + (k - lead length)
    attend_to(h, {XPOS: 1, POS: 1, SEPPOS: -1, ONE: -lead_len, g: -len(words(lx.GREET_PREFIX))})
    move(h, [CH[c] for c in lx.CHARS] + [CONT], list(COPY.values()) + [COPYCONT])


def fill():
    for slots, slot in ((SUBJ, "S"), (OBJ, "O")):
        for e, d in slots.items():
            neuron(L_FILL, {SLOT[slot]: 1, d: 1}, -1, {NEXT[" " + e[1]]: 1})
    for slot, vow in (("AS", VOWS), ("AO", VOWO)):
        neuron(L_FILL, {SLOT[slot]: 1, vow: 1}, -1, {NEXT[" an"]: 1})
        neuron(L_FILL, {SLOT[slot]: 1, vow: -1}, 0, {NEXT[" a"]: 1})
    neuron(L_FILL, {PH1: 1}, 0, {NEXT[XSTART]: 1})
    for c in lx.CHARS:  # copy the next char of the unknown word ...
        neuron(L_FILL, {PHR: 1, COPY[c]: 1, COPYCONT: 1, DOT: -1}, -2, {NEXT[c]: 1})
    neuron(L_FILL, {PHR: 1, COPYCONT: -1, DOT: -1}, 0, {NEXT["."]: 1})  # ... until it ends: "."
    neuron(L_FILL, {PHR: 1, DOT: 1}, -1, {NEXT[EOT]: 1})                 # then end the turn


def lists():
    """Whole lists: "countries in europe", "which animals live in the ocean", "capitals u know"."""
    for t in lx.LIST_TYPES:  # what has the answer said so far? (bag over answer tokens only)
        h = head(1)
        score_keys(h, {AFTER: BIG})
        move(h, [ENT[(t, n)] for n in ENTS[t]], [SAID[(t, n)] for n in ENTS[t]], GAIN_A)
    no_entity = {HAS[TYPE[t]]: -1 for t in lx.TYPES if t != "adj"}  # everyday adjectives don't count
    for t, d in TL.items():  # a plural list noun and no entity at all -> the whole type
        neuron(2, plus({HAS[TN[t]]: 1}, no_entity), 0, {d: 1})
    neuron(2, {SEP: 1, HAS[GREET]: -1}, 0, {LOOPSEP: 1})  # an item comes after the turn start, "!" or ","
    neuron(2, {COMMA: 1}, 0, {LOOPSEP: 1})
    neuron(2, {BANG: 1}, 0, {LOOPSEP: 1})
    tn = [HAS[TN[t]] for t in lx.LIST_TYPES]
    neuron(2, dict.fromkeys(tn, 1), 0, {TNANY: 1})  # any list noun at all (clamped)
    neuron(2, dict.fromkeys(tn, 1), -1, {TNANY: -1})
    for e in LIST_ENTS:
        neuron(2, {SAID[e]: 1}, 0, {SAIDC[e]: 1})
        neuron(2, {SAID[e]: 1}, -1, {SAIDC[e]: -1})
    for rel, d in LISTF.items():  # filter = (rel = mentioned object), no subject; chosen by relation or by a list noun
        ts, to, _ = lx.RELATIONS[rel]
        f = {REL[rel]: 1, TNANY: 1, HAS[TYPE[to]]: 2, HAS[TYPE[ts]]: -2}
        neuron(L_HOP, f, -2, {d: 1})
        neuron(L_HOP, f, -3, {d: -1})
    for rel, s, o in FACTS:  # the same facts read backwards, many at once: every s with (s, rel, o)
        if rel in LISTF and s in CAND:
            neuron(L_FACT, {LAST[lx.RELATIONS[rel][1]][o[1]]: 1, LISTF[rel]: 1}, -1, {CAND[s]: 1})
    for d in LISTF.values():
        neuron(L_FACT, {d: 1}, 0, {LISTREL: 1})
    for e, d in CAND.items():
        neuron(L_FACT, {TL[e[0]]: 1}, 0, {d: 1})
    for rel, s, o in FACTS:  # 2 steps: "cities in europe" -> each candidate country's capital
        if rel in ATTR_RELS and o in CAND:
            neuron(L_CMP, {CAND[s]: 1, HAS[TN[ATTR_RELS[rel]]]: 1}, -1, {CAND[o]: 1})
    listing = plus({LISTREL: 1, XFLAG: -2}, dict.fromkeys(TL.values(), 1))  # an unknown word -> not a list
    neuron(L_CMP, listing, 0, {LISTANY: 1})
    neuron(L_CMP, listing, -1, {LISTANY: -1})
    wants_attr = {HAS[TN[t]]: -100 for t in set(ATTR_RELS.values())}
    for e in LIST_ENTS:  # candidate (0/1), not said yet; countries step aside when their attribute was asked for
        r = plus({CAND[e]: 1, SAIDC[e]: -100}, wants_attr if e[0] == "country" else {})
        neuron(L_FLAG, r, 0, {REMAIN[e]: 1})
        neuron(L_FLAG, r, -1, {REMAIN[e]: -1})
    for t in lx.LIST_TYPES:  # keep only the first one left
        es = [(t, n) for n in ENTS[t]]
        for i, e in enumerate(es):
            neuron(L_MODE, {REMAIN[e]: 1, **{REMAIN[x]: -1 for x in es[:i]}}, 0, {PICK[e]: 1})
    left = dict.fromkeys(REMAIN.values(), 1)
    neuron(L_MODE, left, 0, {ANYREM: 1})
    neuron(L_MODE, left, -1, {ANYREM: -1})
    neuron(L_MODE, {LISTANY: 1, FOUND: -1}, 0, {ML: 1})
    sequence([({ML: 1}, 0), ({HAS[GREET]: 1}, 0)], [token_out(t) for t in words(lx.GREET_PREFIX)])
    for e, d in PICK.items():  # say it ...
        neuron(L_FILL, {d: 1, ML: 1, LOOPSEP: 1}, -2, {NEXT[" " + e[1]]: 1})
    neuron(L_FILL, {ML: 1, ISENT: 1, ANYREM: 1}, -2, {NEXT[","]: 1})       # ... then "," while any are left
    neuron(L_FILL, {ML: 1, ISENT: 1, ANYREM: -1}, -1, {NEXT["."]: 1})      # ... else "."
    neuron(L_FILL, {ML: 1, DOT: 1}, -1, {NEXT[EOT]: 1})
    neuron(L_FILL, {ML: 1, LOOPSEP: 1, ANYREM: -1}, -1, {NEXT[EOT]: 1})   # nothing matched


def pack():
    out = dict(wte=wte, wpe=wpe, wu=wu)
    for l in range(N_LAYER):
        heads = attn[l]
        for i, key in enumerate(("wq", "wk", "wv")):
            out[f"{key}{l}"] = np.array([h[i] for h in heads]).reshape(len(heads), D, D_HEAD)
        out[f"wo{l}"] = np.array([h[3] for h in heads]).reshape(len(heads), D_HEAD, D)
        f = len(mlp[l])
        w1, b1, w2 = np.zeros((D, f)), np.zeros(f), np.zeros((f, D))
        for j, (ins, bias, outs) in enumerate(mlp[l]):
            w1[list(ins), j], b1[j], w2[j, list(outs)] = list(ins.values()), bias, list(outs.values())
        out.update({f"w1{l}": w1, f"b1{l}": b1, f"w2{l}": w2})
    return {k: a.astype(np.float32) for k, a in out.items()}


def main():
    embeddings()
    layer0()
    layer1()
    layers2_3()
    hop()
    facts()
    checks()
    frames()
    fill()
    lists()
    weights = pack()
    path = os.path.join(HERE, "weights.npz")
    np.savez_compressed(path, vocab=np.array(VOCAB), dims=np.array(NAMES), n_layer=N_LAYER, fact_layer=L_FACT, **weights)
    nz = sum(int(np.count_nonzero(a)) for a in weights.values())
    print(f"wrote {path}: {len(VOCAB)} tokens, {len(FACTS)} facts, d_model {D}, {nz:,} nonzero weights")


if __name__ == "__main__":
    main()
