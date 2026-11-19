"""Training text for the tiny trained model: hello-llm's facts as question -> answer pairs.

Answers use hello-llm's own sentence patterns (lexicon.FRAMES), so the trained model and the
hand-written one can be compared word for word. The last question pattern of every relation
with 2+ patterns is held out of training, to test whether training generalizes to new wording.
"""
import json
import os

import lexicon as lx

HERE = os.path.dirname(os.path.abspath(__file__))

QUESTIONS = {  # {s} subject, {o} object, {a} "a"/"an" for the subject; last one is held out
    "capital": ["what is the capital of {s}?", "{s}'s capital?", "which country has {o} as its capital?",
                "capital of {s}"],
    "continent": ["what continent is {s} in?", "where is {s}?", "which continent is {s} on?"],
    "language": ["what language do they speak in {s}?", "language of {s}?", "what is spoken in {s}?"],
    "currency": ["what currency does {s} use?", "currency of {s}?", "what money do they use in {s}?"],
    "sound": ["what sound does {a} {s} make?", "which animal says {o}?", "what does {a} {s} say?"],
    "legs": ["how many legs does {a} {s} have?", "legs of {a} {s}?"],
    "class": ["what kind of animal is {a} {s}?", "what is {a} {s}?"],
    "habitat": ["where does {a} {s} live?", "home of {a} {s}?"],
    "diet": ["what does {a} {s} eat?", "diet of {a} {s}?"],
    "opposite": ["what is the opposite of {s}?", "opposite of {s}"],
}


def load_facts():
    """(relation, (type, subject), (type, object)) for every fact; same order as build.load_facts."""
    doc = json.load(open(os.path.join(HERE, "facts.json")))
    facts = []
    for c in doc["countries"]:
        s = ("country", c["country"])
        facts += [("capital", s, ("city", c["capital"])), ("continent", s, ("continent", c["continent"])),
                  ("language", s, ("language", c["language"]))]
        if c.get("currency"):
            facts.append(("currency", s, ("currency", c["currency"])))
    for a in doc["animals"]:
        s = ("animal", a["animal"])
        if a["sound"]:
            facts.append(("sound", s, ("sound", a["sound"])))
        facts += [("legs", s, ("number", str(a["legs"]))), ("class", s, ("class", a["class"])),
                  ("habitat", s, ("habitat", a["habitat"])), ("diet", s, ("diet", a["diet"]))]
    for x, y in doc["opposites"]:
        facts += [("opposite", ("adj", x), ("adj", y)), ("opposite", ("adj", y), ("adj", x))]
    return facts


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def answer_tokens(rel, s, o):
    """hello-llm's sentence for the fact, as hello-llm tokens: ' the', ' capital', ..., '.'"""
    fill = {"S": s, "O": o, "AS": article(s), "AO": article(o)}
    return [w if w in lx.PUNCT else " " + fill.get(w, w) for w in lx.FRAMES[rel].split()]


def examples():
    """Every (fact, question pattern) pair; 'held_out' marks the patterns kept out of training."""
    out = []
    for i, (rel, (_, s), (_, o)) in enumerate(load_facts()):
        patterns = QUESTIONS[rel]
        for j, pat in enumerate(patterns):
            out.append({"fact": i, "rel": rel, "s": s, "o": o, "pattern": j,
                        "q": pat.format(s=s, o=o, a=article(s)), "answer": answer_tokens(rel, s, o),
                        "held_out": len(patterns) > 1 and j == len(patterns) - 1})
    return out
