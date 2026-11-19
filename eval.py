"""Score hello-llm on questions written by another model (heldout.json). Run: python eval.py [file]

A reply passes when it contains the expected word and, for yes/no questions, starts with the right yes/no.
"""
import json
import os
import re
import sys

from model import HERE, load, reply


def expected_list(item, facts):
    """List questions are graded against facts.json itself: exactly the matching set, nothing missing or extra."""
    rows = [(c, "country") for c in facts["countries"]] + [(a, "animal") for a in facts["animals"]]
    if item["relation"]:
        if item["list_of"] in ("capital", "language", "currency", "continent"):  # 2 steps: an attribute of matches
            return {str(c[item["list_of"]]) for c in facts["countries"]
                    if str(c.get(item["relation"])) == str(item["value"]) and c.get(item["list_of"])}
        return {str(r[t]) for r, t in rows if t == item["list_of"] and str(r.get(item["relation"])) == str(item["value"])}
    field = {"country": "country", "capital": "capital", "language": "language", "continent": "continent",
             "currency": "currency", "animal": "animal"}[item["list_of"]]
    return {str(r[field]) for r, _ in rows if r.get(field) is not None}


def passes(got, item, facts=None):
    if "list_of" in item:
        said = got.removeprefix("hello world! ").rstrip(".")
        return set(said.split(", ")) == expected_list(item, facts)
    answer = str(item["answer"]).lower()
    if not re.search(rf"(^|\W){re.escape(answer)}(\W|$)", got):
        return False
    return item["yn"] is None or got.removeprefix("hello world! ").startswith(item["yn"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "heldout.json")
    doc, w = json.load(open(path)), load()
    facts = json.load(open(os.path.join(HERE, "facts.json")))
    total = ok = 0
    misses = []
    for group in (k for k, v in doc.items() if isinstance(v, list)):
        items = doc[group]
        hits = 0
        for item in items:
            got = reply(w, item["q"])
            if passes(got, item, facts):
                hits += 1
            else:
                misses.append((group, item, got))
        print(f"{group:18s} {hits:3d}/{len(items):3d}  {100 * hits / len(items):5.1f}%")
        total, ok = total + len(items), ok + hits
    print(f"{'TOTAL':18s} {ok:3d}/{total:3d}  {100 * ok / total:5.1f}%")
    for group, item, got in misses:
        want = sorted(expected_list(item, facts)) if "list_of" in item else repr(item["answer"]) + (
            ", " + item["yn"] if item["yn"] else "")
        print(f"  ✗ [{group}] {item['q']!r} -> {got!r}  (want {want})")


if __name__ == "__main__":
    main()
