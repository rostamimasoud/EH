#!/usr/bin/env python3
"""Report progress across the book, one line per chapter.

Counts prose words the way a reader would: LaTeX commands, mathematics,
comments and float bodies are stripped out first, so a chapter full of tables
does not flatter itself.

Usage, from EH_Book/:
    python3 tools/status.py
    python3 tools/status.py --short    totals only
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)

TARGETS = {
    "ch01-whole-earth": 5500, "ch02-single-sphere": 6000, "ch03-anatomy": 5000,
    "ch04-atmosphere": 5500, "ch05-ocean": 6000, "ch06-hydrosphere": 5500,
    "ch07-solid-earth": 5000, "ch08-climate-core": 5500, "ch09-calibration": 6000,
    "ch10-composite-hazard": 5500, "ch11-habitable-area": 5000,
    "ch12-groundwater": 12000, "ch13-where": 5500, "ch14-when": 5000,
    "ch15-pathway": 5000, "ch16-does-it-hold": 6000, "ch17-limits": 5500,
    "ch18-living": 4500,
    "appA-mathematics": 3000, "appB-data": 2500, "appC-reproducing": 2000,
}

WORDS_PER_PAGE = 450.0


def prose_words(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    text = re.sub(r"(?<!\\)%.*", "", text)
    for env in ("figure", "table", "tabular", "equation", "align", "verbatim",
                "tikzpicture", "lstlisting"):
        text = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                      " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z@]+\*?\s*(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}\\&~^_]", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z'’]*", text))


def figures_used(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return len(re.findall(r"\\includegraphics", text)) + \
        len(re.findall(r"\\begin\{tikzpicture\}", text))


def bar(fraction, width=18):
    filled = max(0, min(width, int(round(fraction * width))))
    return "#" * filled + "." * (width - filled)


def main():
    short = "--short" in sys.argv
    paths = sorted(glob.glob(os.path.join(BOOK, "chapters", "*.tex"))) + \
        sorted(glob.glob(os.path.join(BOOK, "backmatter", "app*.tex")))

    total_words = 0
    total_target = 0
    total_figs = 0
    written = 0

    if not short:
        print("%-24s %7s %7s  %-18s %5s  %s"
              % ("file", "words", "target", "progress", "figs", "state"))
        print("-" * 78)

    for path in paths:
        slug = os.path.basename(path)[:-4]
        target = TARGETS.get(slug, 0)
        with open(path, encoding="utf-8") as fh:
            head = fh.read(400)
        stub = "STATUS: stub" in head

        words = 0 if stub else prose_words(path)
        figs = 0 if stub else figures_used(path)
        total_words += words
        total_target += target
        total_figs += figs
        if not stub:
            written += 1

        if not short:
            frac = (words / float(target)) if target else 0.0
            state = "stub" if stub else ("ok" if frac >= 0.9 else "short")
            print("%-24s %7d %7d  %-18s %5d  %s"
                  % (slug, words, target, bar(frac), figs, state))

    print("-" * 78)
    print("%-24s %7d %7d  %-18s %5d  %d of %d written"
          % ("TOTAL", total_words, total_target,
             bar(total_words / float(total_target) if total_target else 0),
             total_figs, written, len(paths)))
    print("")
    print("estimated pages of prose: %d   (front and back matter add about 40)"
          % round(total_words / WORDS_PER_PAGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
