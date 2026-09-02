#!/usr/bin/env python3
"""Check every citation in the book.

Three independent checks:

  1. KEYS      Every key cited in a .tex file exists in references.bib, and
               every entry in references.bib is actually cited somewhere.
  2. RESOLVE   Every doi field resolves at doi.org.
  3. METADATA  The title and year registered against that DOI agree with what
               the .bib entry claims. This is the check that catches a copied
               DOI pointing at the wrong paper, which is the failure mode a
               human proofreader almost never spots.

Results are cached in tools/.citecache.json keyed by DOI, so repeat runs cost
nothing and need no network. Delete the cache to force a full recheck.

Usage, from EH_Book/:
    python3 tools/checkcites.py            # cached where possible
    python3 tools/checkcites.py --refresh  # ignore the cache
    python3 tools/checkcites.py --offline  # keys only, no network
"""

import concurrent.futures
import difflib
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
BIB = os.path.join(BOOK, "references.bib")
CACHE = os.path.join(HERE, ".citecache.json")

TEX_GLOBS = ["chapters/*.tex", "frontmatter/*.tex", "backmatter/*.tex", "main.tex"]

ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")
FIELD = re.compile(r"^\s*(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", re.MULTILINE | re.DOTALL)
CITE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")

TITLE_MATCH_FLOOR = 0.72

# Mismatches that have been examined by hand and are cosmetic. The value is the
# reason, printed so the exemption stays visible and does not become a way of
# quietly silencing a real problem.
ACCEPTED = {
    "Forster2021":
        "typographic apostrophe and a comma; registered year 2023 is the "
        "reprint of the 2021 assessment chapter",
    "ipcc2022ar6wg2":
        "entry carries the chapter number as part of the title, the registry "
        "carries the bare chapter name",
}

# Mismatches that are real and unresolved. Listed so they are reported loudly
# every run until someone identifies the intended work. See corrections.bib.
UNRESOLVED = {
    "bekaert2017centralvalley":
        "doi belongs to a different paper and no work with this title exists "
        "in Crossref under this author",
    "sneed2008extensometer":
        "doi belongs to a Kansas water quality report; the intended USGS "
        "subsidence report has not been identified",
}


def strip_braces(text):
    return re.sub(r"[{}\\]", "", text).strip()


def parse_bib(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    entries = {}
    for match in ENTRY_START.finditer(text):
        etype, key = match.group(1).lower(), match.group(2)
        if etype in ("comment", "preamble", "string"):
            continue
        depth, end = 0, None
        for i in range(text.index("{", match.start()), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        body = text[match.end():end]
        fields = {}
        for fmatch in FIELD.finditer(body):
            fields[fmatch.group(1).lower()] = strip_braces(fmatch.group(2))
        entries[key] = {"type": etype, "fields": fields}
    return entries


def cited_keys():
    used = {}
    for pattern in TEX_GLOBS:
        for path in sorted(glob.glob(os.path.join(BOOK, pattern))):
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if line.lstrip().startswith("%"):
                        continue
                    for match in CITE.finditer(line):
                        for key in match.group(1).split(","):
                            key = key.strip()
                            if key:
                                used.setdefault(key, []).append(
                                    "%s:%d" % (os.path.relpath(path, BOOK), lineno))
    return used


def fetch(doi):
    """Ask doi.org for CSL JSON metadata. Returns dict or an error marker."""
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/:()<>;-._")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "EH-Book-citation-check/1.0 (mailto:rostamimasoud@yahoo.com)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return {"error": "HTTP %d" % exc.code}
    except Exception as exc:                                  # noqa: BLE001
        return {"error": type(exc).__name__}

    title = payload.get("title")
    if isinstance(title, list):
        title = title[0] if title else ""
    year = ""
    for field in ("issued", "published-print", "published-online", "created"):
        parts = payload.get(field, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    return {
        "title": title or "",
        "year": year,
        "container": payload.get("container-title") or "",
        "type": payload.get("type", ""),
    }


def normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def main():
    refresh = "--refresh" in sys.argv
    offline = "--offline" in sys.argv

    entries = parse_bib(BIB)
    used = cited_keys()
    problems = 0

    print("=" * 68)
    print("1. KEYS")
    print("=" * 68)

    undefined = sorted(k for k in used if k not in entries)
    for key in undefined:
        problems += 1
        print("  UNDEFINED  %-34s cited at %s" % (key, ", ".join(used[key][:3])))
    if not undefined:
        print("  ok         every cited key exists in references.bib")

    uncited = sorted(k for k in entries if k not in used)
    print("  note       %d of %d entries are not yet cited (expected while "
          "chapters are stubs)" % (len(uncited), len(entries)))

    if offline:
        print("\noffline mode: skipping DOI resolution and metadata check")
        return 1 if problems else 0

    cache = {}
    if os.path.exists(CACHE) and not refresh:
        try:
            with open(CACHE, encoding="utf-8") as fh:
                cache = json.load(fh)
        except ValueError:
            cache = {}

    with_doi = {k: v["fields"]["doi"] for k, v in entries.items()
                if v["fields"].get("doi")}
    todo = sorted({d for d in with_doi.values() if d not in cache})

    if todo:
        print("\n  resolving %d new DOIs at doi.org ..." % len(todo))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for doi, result in zip(todo, pool.map(fetch, todo)):
                cache[doi] = result
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)

    print("")
    print("=" * 68)
    print("2. RESOLVE")
    print("=" * 68)

    dead = []
    for key in sorted(with_doi):
        doi = with_doi[key]
        result = cache.get(doi, {})
        if result.get("error"):
            dead.append((key, doi, result["error"]))
    for key, doi, err in dead:
        problems += 1
        print("  DEAD       %-30s %s  (%s)" % (key, doi, err))
    if not dead:
        print("  ok         all %d DOIs resolve" % len(with_doi))

    nodoi = sorted(k for k in entries if not entries[k]["fields"].get("doi"))
    print("  note       %d entries carry no DOI:" % len(nodoi))
    for key in nodoi:
        year = entries[key]["fields"].get("year", "?")
        print("               %-34s %s  %s" % (key, year, entries[key]["type"]))

    print("")
    print("=" * 68)
    print("3. METADATA")
    print("=" * 68)

    mismatched = []
    for key in sorted(with_doi):
        doi = with_doi[key]
        result = cache.get(doi, {})
        if result.get("error"):
            continue
        claimed = entries[key]["fields"].get("title", "")
        actual = result.get("title", "")
        ratio = difflib.SequenceMatcher(
            None, normalise(claimed), normalise(actual)).ratio()

        claimed_year = entries[key]["fields"].get("year", "").strip()
        actual_year = result.get("year", "").strip()
        year_off = None
        if claimed_year.isdigit() and actual_year.isdigit():
            delta = abs(int(claimed_year) - int(actual_year))
            if delta > 1:
                year_off = (claimed_year, actual_year)

        if ratio < TITLE_MATCH_FLOOR or year_off:
            mismatched.append((key, doi, ratio, claimed, actual, year_off))

    for key, doi, ratio, claimed, actual, year_off in mismatched:
        if key in ACCEPTED:
            print("  accepted   %-30s %s" % (key, ACCEPTED[key]))
            continue
        problems += 1
        label = "UNRESOLVED" if key in UNRESOLVED else "MISMATCH  "
        print("  %s %s" % (label, key))
        if key in UNRESOLVED:
            print("               %s" % UNRESOLVED[key])
        print("               doi       %s" % doi)
        print("               bib says  %s" % (claimed[:88] or "(no title)"))
        print("               doi says  %s" % (actual[:88] or "(none returned)"))
        print("               title similarity %.2f" % ratio)
        if year_off:
            print("               year      bib %s, registered %s"
                  % year_off)
    if not [m for m in mismatched if m[0] not in ACCEPTED]:
        print("  ok         every DOI points at a work whose title and year "
              "match the entry")

    print("")
    print("=" * 68)
    print("%d problem(s)" % problems)
    print("=" * 68)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
