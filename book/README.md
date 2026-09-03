# Book scripts

Supporting code for the monograph *Earth Habitability: A multi sphere
perspective on our changing planet* (Rostami and Fu).

The book itself is written against the results produced by the `eh_shallow`
package on the `main` branch of this repository. This branch carries only the
code needed to reproduce the book's figures and to check its text, so that the
manuscript directory holds prose and nothing else.

## Contents

`springerbook.sty`
: Springer monograph page style built on the standard `book` class: 155 by 235
  mm trim, a 117.5 mm measure, Springer chapter openers, running heads, chapter
  abstracts and highlight boxes. Springer distributes `svmono.cls` only from its
  own author pages and it is not on CTAN, so this reproduces the metrics. The
  master file carries a commented pair of lines that switches to the genuine
  class without any change to the chapter sources.

`figtools/bookstyle.py`
: Figure style for the book. A single column measure, a serif face matching the
  body text, and a palette in which each sphere keeps one colour throughout.
  Deliberately distinct from the two column journal style used for the papers,
  so that no figure in the book is a reproduction of a published one.

`figtools/make_extra.py`
: Recomputes the quantities the stored results archive does not carry, and caches
  them in one file for the figure code. It reruns the calibration to recover the
  posterior particle cloud, runs the emulator under every pathway to recover the
  six carried variables including the ocean chemistry, evaluates the composite
  hazard field at four dates under two pathways, computes the same field with the
  water hazard term switched off, and derives the time of emergence maps and the
  threshold sensitivity curve. Deterministic: same seed, same output.

`figtools/make_figs.py`
: Renders every data figure in the book, from `outputs/results.npz`,
  `outputs/metrics.json`, the released water hazard field and the recomputed
  archive above. Same science as the published work, redrawn at book measure with
  new layouts and colour scales. Includes a full page composition of the water
  hazard field with regional detail.

`figures/tikz/`
: Conceptual diagrams drawn for the book: the column through a single place, the
  tiered construction of the composite score, recoverable against unrecoverable
  compaction, and an original rendering of a planetary boundary assessment.

`tools/merge_bib.py`
: Merges the two source bibliographies, reports key collisions instead of
  resolving them silently, and applies a corrections overlay.

`tools/checkcites.py`
: Verifies citations three ways. Every cited key must exist. Every DOI must
  resolve. Every DOI must point at a work whose registered title and year match
  the entry, which is what catches an identifier copied from the wrong paper.
  Results are cached, so repeat runs need no network.

`tools/stylecheck.py`
: Enforces the book's writing rules mechanically: no hyphens or dashes in prose,
  no lists, no reference to code, a bounded citation density, and the closing
  cross sphere summary that every chapter carries.

`tools/status.py`
: Prose word count per chapter against target, with an estimated page count.

`tools/build.sh`
: Runs the full LaTeX chain and reports only errors, undefined references, bad
  boxes and the page count.

## Use

From the book directory:

    python3 figtools/make_figs.py --all
    python3 tools/checkcites.py
    python3 tools/stylecheck.py
    bash tools/build.sh

## Requirements

Python with NumPy and Matplotlib for the figures. A TeX distribution with
`natbib`, `doi`, `hyperref`, `titlesec`, `tcolorbox`, `newtx` and `tikz` for the
build. No Springer class file is required.
