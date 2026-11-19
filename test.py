"""Checks for hello-llm v2. Run: python build.py && python test.py

1 sanity   my own questions, must all pass
2 ablation zero the one (peru, capital, lima) neuron: every phrasing of that fact breaks, others don't
3 audit    no neuron maps question words straight to an answer word
"""
import ast

import numpy as np

from model import compact, load, reply

CASES = [
    ("hello", "hello world!"),
    ("hi there", "hello world!"),
    ("what is the capital of peru?", "the capital of peru is lima."),
    ("Peru's capital?", "the capital of peru is lima."),
    ("which country has lima as its capital?", "the capital of peru is lima."),
    ("lima is the capital of which country", "the capital of peru is lima."),
    ("is lima the capital of chile?", "no, the capital of chile is santiago."),
    ("is lima the capital of peru?", "yes, the capital of peru is lima."),
    ("what continent is japan in?", "japan is in asia."),
    ("is poland in europe?", "yes, poland is in europe."),
    ("is brazil in africa", "no, brazil is in south america."),
    ("what language do they speak in brazil?", "people in brazil speak portuguese."),
    ("do they speak spanish in mexico?", "yes, people in mexico speak spanish."),
    ("what sound does a cow make?", "a cow says moo."),
    ("what does an owl say", "an owl says hoot."),
    ("which animal says moo?", "a cow says moo."),
    ("how many legs does a spider have?", "a spider has 8 legs."),
    ("do cats have four legs?", "yes, a cat has 4 legs."),
    ("is a whale a fish?", "no, a whale is a mammal."),
    ("what kind of animal is a frog", "a frog is an amphibian."),
    ("tell me about elephants", "an elephant is a mammal."),
    ("what is the opposite of hot?", "the opposite of hot is cold."),
    ("opposite of cold", "the opposite of cold is hot."),
    ("hi, what is the capital of france?", "hello world! the capital of france is paris."),
    ("whats the capital of the usa", "the capital of united states is washington."),
    ("what continent is tokyo in?", "japan is in asia."),                      # 2 hops: city -> country -> fact
    ("is ottawa in europe?", "no, canada is in north america."),
    ("what language do they speak in paris", "people in france speak french."),
    ("mexico's language is spanish?", "yes, people in mexico speak spanish."),  # a claim to check, no is/are
    ("athens is greece's capital, isn't it?", "yes, the capital of greece is athens."),
    ("black's opposite white, right?", "yes, the opposite of black is white."),
    ("which animal barks?", "a fox says bark."),
    ("where do they speak portuguese", "portugal, brazil, angola, mozambique."),  # lists: many facts at once
    ("which animals live in the ocean", "dolphin, whale, seagull, seal, penguin."),
    ("continents u know", "africa, asia, europe, north america, south america, oceania."),
    ("currencies used in south america", "dollar, peso, brazilian real, sol, bolivar, guarani."),  # 2-step list
    ("what continent is south africa in", "i don't know about south."),
    ("what currency does japan use", "the currency of japan is the yen."),
    ("is a lion a herbivore?", "no, a lion is a carnivore."),
    ("where do penguins live", "the home of a penguin is the ocean."),
    ("what is the capital of wakanda?", "i don't know about wakanda."),
    ("how are you", "i don't know."),
    ("", "i don't know."),
]
PERU = ["what is the capital of peru?", "which country has lima as its capital?", "is lima the capital of peru?"]
OTHERS = [("what is the capital of chile?", "the capital of chile is santiago."),
          ("what language do they speak in peru?", "people in peru speak spanish.")]


def dims(w, prefix):
    return [i for i, n in enumerate(w["dims"]) if str(n).startswith(prefix)]


def sanity(w):
    bad = [(m, want, got) for m, want in CASES if (got := reply(w, m)) != want]
    for m, want, got in bad:
        print(f"  ✗ {m!r}\n      want {want!r}\n      got  {got!r}")
    print(f"sanity   {len(CASES) - len(bad)}/{len(CASES)}")
    return not bad


def ablation(w):
    l = int(w["fact_layer"])
    peru, lima = dims(w, "subj:('country', 'peru')"), dims(w, "obj:('city', 'lima')")
    hits = [j for j in range(w[f"w2{l}"].shape[0]) if w[f"w2{l}"][j, peru[0]] and w[f"w2{l}"][j, lima[0]]]
    assert len(hits) == 1, hits
    cut = dict(w, **{f"w2{l}": w[f"w2{l}"].copy()})
    cut[f"w2{l}"][hits[0]] = 0
    cut["fast"] = [compact(cut, i) for i in range(int(w["n_layer"]))]
    broken = [m for m in PERU if "lima" not in reply(cut, m)]
    intact = [m for m, want in OTHERS if reply(cut, m) == want]
    ok = len(broken) == len(PERU) and len(intact) == len(OTHERS)
    print(f"ablation neuron L{l}#{hits[0]} zeroed: {len(broken)}/{len(PERU)} peru phrasings lose 'lima', "
          f"{len(intact)}/{len(OTHERS)} other facts intact  {'✓' if ok else '✗'}")
    return ok


def audit(w):
    """Every neuron that votes for an entity word must read a computed slot (subj/obj), never question words."""
    names = [str(n) for n in w["dims"]]
    entity_words = {" " + ast.literal_eval(n[4:])[1] for n in names if n.startswith("ent:")}
    entity_next = {i for i, n in enumerate(names) if n.startswith("next:") and ast.literal_eval(n[5:]) in entity_words}
    question_side = set(dims(w, "ent:") + dims(w, "first-") + dims(w, "last-") + dims(w, "cue:"))
    slots = set(dims(w, "subj:") + dims(w, "obj:") + dims(w, "pick:"))  # computed: fact output / list pick
    bad = 0
    for l in range(int(w["n_layer"])):
        w1, w2 = w[f"w1{l}"], w[f"w2{l}"]
        for j in range(w2.shape[0]):
            writes = set(np.nonzero(w2[j])[0])
            if writes & entity_next:
                reads = set(np.nonzero(w1[:, j])[0])
                bad += bool(reads & question_side) or not (reads & slots) or len(writes) > 1
    print(f"audit    {bad} neurons map question words straight to an answer word  {'✓' if not bad else '✗'}")
    return not bad


def main():
    w = load()
    results = [sanity(w), ablation(w), audit(w)]
    assert all(results)


if __name__ == "__main__":
    main()
