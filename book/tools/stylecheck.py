#!/usr/bin/env python3
"""Enforce the writing rules in STYLE.md.

Checks the prose stream only. Everything that is not prose is masked out first:
comments, mathematics, labels, references, citation keys, graphics paths, web
addresses, index entries, environment names and macro names. Masking is what
makes the hyphen rule usable at all, since hyphens are legitimate and
unavoidable inside a DOI or a file path.

Rules enforced, keyed to STYLE.md:
    1  no hyphen or dash characters in prose
    2  no "X, rather than Y" and no "X, not Y"
    3  no itemize, enumerate or description environments in chapters
    4  no deep time vocabulary
    5  no reference to a script, module or function
    7  citation density stays light
    8  every chapter closes with a spherebox

Usage, from EH_Book/:
    python3 tools/stylecheck.py                 # all chapters and front matter
    python3 tools/stylecheck.py chapters/ch12-groundwater.tex
    python3 tools/stylecheck.py --quiet         # counts only
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)

DEFAULT_GLOBS = ["chapters/*.tex", "frontmatter/*.tex", "backmatter/*.tex"]

# Files exempt from a given rule, with the reason.
EXEMPT = {
    "frontmatter/acronyms.tex": {1, 3, 7, 8},   # a glossary is a lookup table
    "frontmatter/titlepage.tex": {7, 8},
    "frontmatter/dedication.tex": {7, 8},
    "frontmatter/preface.tex": {8},             # the preface has no spherebox
    "backmatter/glossary.tex": {3, 7, 8},
    "backmatter/appA-mathematics.tex": {8},
    "backmatter/appB-data.tex": {8},
    "backmatter/appC-reproducing.tex": {5, 8},  # this appendix gives the repository
}

# Proper nouns whose hyphen is part of the name and cannot be removed. These are
# masked before the hyphen rule runs. Keep this list short and specific.
ALLOWED_HYPHENS = [
    "Li-Yun",
    "Rostami-",
    "GRACE-FO",
    "Sentinel-1",
    "ALOS-2",
    "e-folding",
]

DASHES = {
    "-": "hyphen",
    "\u2010": "Unicode hyphen",
    "\u2011": "non breaking hyphen",
    "\u2012": "figure dash",
    "\u2013": "en dash",
    "\u2014": "em dash",
    "\u2015": "horizontal bar",
}

BANNED_SHAPES = [
    (re.compile(r",\s*rather than\b", re.I), 'the banned "X, rather than Y" shape'),
    (re.compile(r",\s*not\b", re.I), 'the banned "X, not Y" shape'),
]

DEEPTIME = re.compile(
    r"\b(paleo\w*|palaeo\w*|phanerozoic|holocene|pleistocene|cretaceous|"
    r"jurassic|triassic|permian|neogene|quaternary|deep\s+time|"
    r"geological\s+past|faint\s+young\s+sun|habitable\s+zone|"
    r"circumstellar|exoplanet\w*)\b", re.I)

CODEREF = re.compile(
    r"(\.py\b|\.sh\b|\.npz\b|\.json\b|\bpython\b|\bnumpy\b|\bscipy\b|"
    r"\bmatplotlib\b|\bscikit\b|\bsklearn\b|\bgit\s?hub\s+repository\s+at\b|"
    r"\w+\(\)|\b\w+\.py\b|--[a-z-]+\b)", re.I)

LIST_ENV = re.compile(r"\\begin\{(itemize|enumerate|description)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")
STRAIGHT_QUOTE = re.compile(r'(?<!\\)"')


def mask(line):
    """Blank out everything that is not prose, preserving column positions."""
    out = list(line)

    def blank(match, group=0):
        for i in range(match.start(group), match.end(group)):
            out[i] = "\x00"

    text = line

    # Proper nouns whose hyphen is part of the name.
    for name in ALLOWED_HYPHENS:
        for match in re.finditer(re.escape(name), text):
            blank(match)
        text = re.sub(re.escape(name), "\x00" * len(name), text)

    # Comments: everything after an unescaped percent sign.
    comment = re.search(r"(?<!\\)%", text)
    if comment:
        for i in range(comment.start(), len(out)):
            out[i] = "\x00"
        text = text[:comment.start()] + "\x00" * (len(text) - comment.start())

    # Inline and display mathematics.
    for pattern in (r"\$\$.*?\$\$", r"(?<!\\)\$.*?(?<!\\)\$",
                    r"\\\[.*?\\\]", r"\\\(.*?\\\)"):
        for match in re.finditer(pattern, text, re.S):
            blank(match)

    # Arguments that are identifiers and never prose.
    for pattern in (
        r"\\(?:label|ref|eqref|pageref|cite[a-zA-Z]*|index|url|nolinkurl|doi|"
        r"includegraphics|input|include|bibliography|bibliographystyle|"
        r"hypersetup|usepackage|documentclass|graphicspath|addcontentsline)"
        r"\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
        r"\\href\s*\{[^{}]*\}",
        r"\\begin\s*\{[^{}]*\}", r"\\end\s*\{[^{}]*\}",
        r"\\[a-zA-Z@]+\*?",                       # macro names themselves
        r"\\verb\|[^|]*\|",
        r"\\texttt\s*\{[^{}]*\}",                 # code spans are not prose
    ):
        for match in re.finditer(pattern, text):
            blank(match)

    return "".join(out)


def check_file(path, quiet=False):
    rel = os.path.relpath(path, BOOK)
    exempt = EXEMPT.get(rel, set())
    problems = []

    with open(path, encoding="utf-8") as fh:
        raw = fh.readlines()

    in_verbatim = False
    in_math = False
    body = []
    math_open = re.compile(
        r"\\begin\{(equation|align|gather|multline|eqnarray|array|split|"
        r"cases|displaymath|tikzpicture)\*?\}")
    math_close = re.compile(
        r"\\end\{(equation|align|gather|multline|eqnarray|array|split|"
        r"cases|displaymath|tikzpicture)\*?\}")

    for lineno, line in enumerate(raw, 1):
        if re.search(r"\\begin\{(verbatim|lstlisting|Verbatim)\}", line):
            in_verbatim = True
        if in_verbatim:
            if re.search(r"\\end\{(verbatim|lstlisting|Verbatim)\}", line):
                in_verbatim = False
            continue
        # Display mathematics spans several lines, so it has to be tracked as
        # state. A minus sign inside an equation is arithmetic and not prose.
        if math_open.search(line):
            in_math = True
            body.append((lineno, ""))
            continue
        if in_math:
            if math_close.search(line):
                in_math = False
            body.append((lineno, ""))
            continue
        body.append((lineno, line.rstrip("\n")))

    for lineno, line in body:
        prose = mask(line)

        if 1 not in exempt:
            for ch, name in DASHES.items():
                col = prose.find(ch)
                if col >= 0:
                    problems.append((lineno, 1, "%s in prose" % name,
                                     line.strip()[:78]))
                    break

        if 2 not in exempt:
            for pattern, label in BANNED_SHAPES:
                if pattern.search(prose):
                    problems.append((lineno, 2, label, line.strip()[:78]))
                    break

        if 3 not in exempt and LIST_ENV.search(line):
            problems.append((lineno, 3, "list environment; write it as prose",
                             line.strip()[:78]))

        if 4 not in exempt and DEEPTIME.search(prose):
            hit = DEEPTIME.search(prose).group(0)
            problems.append((lineno, 4, "deep time vocabulary: %s" % hit,
                             line.strip()[:78]))

        if 5 not in exempt and CODEREF.search(prose):
            hit = CODEREF.search(prose).group(0)
            problems.append((lineno, 5, "names code: %s" % hit,
                             line.strip()[:78]))

        if STRAIGHT_QUOTE.search(prose):
            problems.append((lineno, 0, "straight double quote; use `` and ''",
                             line.strip()[:78]))

    text = "".join(l for _, l in body)

    if 8 not in exempt and "\\begin{spherebox}" not in text:
        problems.append((0, 8, "chapter has no closing spherebox", ""))

    if 7 not in exempt:
        words = len(re.findall(r"[A-Za-z']+", mask(text)))
        cites = len(CITE.findall(text))
        if words > 400:
            per_page = cites / (words / 450.0)
            if per_page > 6:
                problems.append(
                    (0, 7, "citation density %.1f per page, aim for two or "
                     "three" % per_page, ""))
        for match in CITE.finditer(text):
            keys = [k for k in match.group(1).split(",") if k.strip()]
            if len(keys) > 2:
                problems.append((0, 7, "%d keys stacked in one citation: %s"
                                 % (len(keys), match.group(1)[:50]), ""))

    if not quiet:
        for lineno, rule, message, excerpt in problems:
            where = "%s:%d" % (rel, lineno) if lineno else rel
            print("  %-42s rule %d  %s" % (where, rule, message))
            if excerpt:
                print("  %-42s          %s" % ("", excerpt))

    return problems


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv

    if args:
        paths = [a if os.path.isabs(a) else os.path.join(BOOK, a) for a in args]
    else:
        paths = []
        for pattern in DEFAULT_GLOBS:
            paths.extend(sorted(glob.glob(os.path.join(BOOK, pattern))))

    total = 0
    clean = 0
    for path in paths:
        if not os.path.exists(path):
            print("  missing: %s" % path)
            total += 1
            continue
        problems = check_file(path, quiet)
        if problems:
            total += len(problems)
        else:
            clean += 1

    print("")
    print("%d file(s) checked, %d clean, %d problem(s)"
          % (len(paths), clean, total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
