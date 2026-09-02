#!/usr/bin/env bash
# Build the book and report only what matters.
#
# The full LaTeX log is tens of thousands of lines and must never be read into a
# working session. This script runs the whole chain and prints a short summary:
# errors, undefined references, overfull boxes past a tolerance, and the page
# count. The log stays on disk for anyone who wants it.
#
#   bash tools/build.sh          full chain
#   bash tools/build.sh --fast   one pass, no bibliography, for a quick look

set -eo pipefail

cd "$(dirname "$0")/.."
BOOK=$(pwd)
JOB=main
FAST=0
[ "$1" = "--fast" ] && FAST=1

mkdir -p build
# \include writes an .aux file beside each included source, so the output
# directory needs the same subdirectory layout or pdflatex stops dead.
mkdir -p build/frontmatter build/chapters build/backmatter build/figures

run() {
  pdflatex -interaction=nonstopmode -halt-on-error=false \
           -output-directory=build -file-line-error "$JOB.tex" \
           > /dev/null 2>&1 || true
}

echo "building ..."
run
if [ "$FAST" -eq 0 ]; then
  if [ -f "build/$JOB.aux" ]; then
    # bibtex runs inside build/, so it needs to be told where the .bib and the
    # .bst actually live. Without this it silently produces an empty .bbl.
    ( cd build && BIBINPUTS="$BOOK:$BIBINPUTS" BSTINPUTS="$BOOK:$BSTINPUTS" \
        bibtex "$JOB" > /dev/null 2>&1 || true )
  fi
  if [ -f "build/$JOB.idx" ]; then
    ( cd build && makeindex "$JOB" > /dev/null 2>&1 || true )
  fi
  run
  run
fi

LOG="build/$JOB.log"
BLG="build/$JOB.blg"

echo ""
echo "===================== errors ====================="
if grep -qE "^(.*:[0-9]+:|! )" "$LOG"; then
  grep -E "^(.*:[0-9]+:|! )" "$LOG" | grep -v "^! *$" | sort -u | head -25
else
  echo "  none"
fi

echo ""
echo "================ undefined references ============"
UNDEF=$(grep -oE "(Citation|Reference) \`[^']+' on page [0-9]+ undefined" "$LOG" \
        | sed -E "s/ on page [0-9]+//" | sort -u || true)
if [ -n "$UNDEF" ]; then
  echo "$UNDEF" | head -25
  echo "  ... $(echo "$UNDEF" | wc -l) distinct"
else
  echo "  none"
fi

echo ""
echo "================== bibliography =================="
if [ -f "$BLG" ]; then
  if grep -q "I found no \\\\citation commands" "$BLG"; then
    echo "  no citations yet (expected while chapters are stubs)"
  elif grep -qiE "error|warning" "$BLG"; then
    grep -iE "error|warning" "$BLG" | grep -v "^Warning\$" | sort -u | head -15
  else
    echo "  clean"
  fi
  if [ -f "build/$JOB.bbl" ]; then
    echo "  entries in reference list: $(grep -c '^\\bibitem' "build/$JOB.bbl" || echo 0)"
  fi
else
  echo "  not run"
fi

echo ""
echo "=================== bad boxes ===================="
OVER=$(grep -c "Overfull \\\\hbox" "$LOG" || true)
BADV=$(grep -c "Underfull \\\\vbox" "$LOG" || true)
echo "  overfull hboxes:  ${OVER:-0}"
echo "  underfull vboxes: ${BADV:-0}"
grep "Overfull \\\\hbox" "$LOG" | grep -oE "\(([0-9]+\.[0-9]+)pt too wide\)" \
  | sort -rn -t'(' -k2 | head -3 | sed 's/^/  worst: /' || true

echo ""
echo "===================== result ====================="
if [ -f "build/$JOB.pdf" ]; then
  PAGES=$(pdfinfo "build/$JOB.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  SIZE=$(pdfinfo "build/$JOB.pdf" 2>/dev/null | awk '/^Page size:/{$1="";$2="";print}')
  echo "  build/$JOB.pdf"
  echo "  pages:     ${PAGES:-unknown}  (target about 350)"
  echo "  page size:${SIZE:- unknown}"
else
  echo "  NO PDF PRODUCED"
  exit 1
fi
echo ""
