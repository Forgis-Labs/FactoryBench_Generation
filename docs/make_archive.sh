#!/usr/bin/env bash
# Build a self-contained submission archive for one workshop venue.
#
#   ./make_archive.sh wmphysai         (or: make -C docs archive-wmphysai)
#
# The repo keeps workshop sources split across three directories so that every
# venue shares one body (see ../Makefile). A submission portal cannot be handed
# that: it wants a tree that compiles on its own machine. So this script stages
# a FLAT copy, rewrites the cross-directory paths to local ones, compiles it in
# place to prove the copy is complete, and zips the result.
#
# Output, in docs/dist/:
#   factorybench-<venue>-workshop.pdf         the PDF to upload
#   factorybench-<venue>-workshop-source.zip  flat sources, `pdflatex main` builds
#
# The staged tree is left in dist/<venue>-source/ for inspection.
set -euo pipefail

VENUE="${1:?usage: make_archive.sh <venue>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

WRAPPER="workshop_tex/main_${VENUE}.tex"
CONTENT="paper/_workshop_${VENUE}.tex"
[ -f "$WRAPPER" ] || { echo "no wrapper: $WRAPPER" >&2; exit 1; }
[ -f "$CONTENT" ] || { echo "no venue content file: $CONTENT" >&2; exit 1; }

STAGE="dist/${VENUE}-source"
rm -rf "$STAGE"
mkdir -p "$STAGE/figures"

# Everything staged below is derived from the build's own log, so a venue that
# reads a file the others do not (the one shipping the appendix, say) packages
# correctly without this script knowing which venue is which.
LOG="workshop_tex/main_${VENUE}.log"
[ -f "$LOG" ] || { echo "no build log: $LOG -- run make workshop-${VENUE} first" >&2; exit 1; }

# --- sources -------------------------------------------------------------
# The wrapper becomes main.tex so the archive builds with a bare `pdflatex main`.
cp "$WRAPPER"                       "$STAGE/main.tex"
cp workshop_tex/neurips_2026.sty    "$STAGE/"
cp paper/_preamble_common.tex       "$STAGE/"
cp paper/_titleauthor.tex           "$STAGE/"
cp paper/_workshop_core.tex         "$STAGE/"
cp "$CONTENT"                       "$STAGE/"
cp paper/references.bib             "$STAGE/"

# Any other shared source this venue reads -- _appendix.tex for the venue whose
# CFP exempts it, and whatever a future venue adds.
grep -oE '\.\./paper/[A-Za-z0-9_-]+\.tex' "$LOG" | sort -u | while read -r ref; do
    src="workshop_tex/$ref"
    if [ -f "$src" ]; then cp "$src" "$STAGE/"; fi
done

# --- figures -------------------------------------------------------------
# Only what this build actually reads, derived from its own log rather than a
# hardcoded list: venues differ (the one that ships the appendix pulls in
# figures the others never touch), and a hardcoded list silently rots. An
# archive carrying 25 MB of unreferenced assets is not a clean one.
grep -oE '\.\./[A-Za-z0-9_./-]+\.(png|pdf|jpe?g|tex)' "$LOG" \
  | sort -u \
  | while read -r ref; do
        src="workshop_tex/$ref"
        [ -f "$src" ] || continue
        case "$ref" in
            */figures/*) cp "$src" "$STAGE/figures/" ;;   # graphics and figure .tex
            *)           : ;;                            # shared sources, copied above
        esac
    done

# The appendix pulls figures from ../../output/figures/ rather than the build
# directory; those land in figures/ too and the path rewrite below matches.
# (`[ -f x ] && cp` would return 1 on the last miss and, under set -e, kill the
# script — hence the explicit if.)
grep -oE '\.\./\.\./output/figures/[A-Za-z0-9_.-]+\.(pdf|png)' "$LOG" | sort -u \
  | while read -r ref; do
        src="workshop_tex/$ref"
        if [ -f "$src" ]; then cp "$src" "$STAGE/figures/"; fi
    done

echo ">>> staged $(find "$STAGE/figures" -type f | wc -l) figures for ${VENUE}"

# --- flatten the paths ---------------------------------------------------
# Rewrite the copies only; the repo sources keep their shared layout.
cd "$STAGE"
# \input{../paper/x} -> \input{x}, \bibliography{../paper/references} ->
# {references}, ../../output/figures/y -> figures/y, and the two search paths
# the wrapper points at ../neurips_tex/ -> here.
sed -i \
    -e 's|\.\./paper/||g' \
    -e 's|\.\./\.\./output/figures/|figures/|g' \
    -e 's|\.\./neurips_tex/figures/|./figures/|g' \
    -e 's|\.\./neurips_tex/|./|g' \
    -e 's|\.\./\.\./|./|g' \
    *.tex

# Any surviving parent-directory reference in live LaTeX (not in a comment)
# means the archive would reach outside itself. Comments may mention repo
# paths freely, so only non-comment text counts.
if grep -nE '^[^%]*\.\./' *.tex figures/*.tex 2>/dev/null; then
    echo "FAIL: the staged archive still references a parent directory" >&2
    exit 1
fi

# --- prove it compiles standalone ---------------------------------------
echo ">>> compiling the staged archive to verify it is self-contained"
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

fail=0
if grep -q '^!' main.log; then
    echo "FAIL: LaTeX errors in the staged archive:" >&2
    grep -n '^!' main.log >&2
    fail=1
fi
if grep -qE "(Citation|Reference) \`[^']*' on page [0-9]+ undefined" main.log; then
    echo "FAIL: undefined citations or references:" >&2
    grep -oE "(Citation|Reference) \`[^']*' on page [0-9]+ undefined" main.log | sort -u >&2
    fail=1
fi
if grep -qE "File \`[^']*' not found" main.log; then
    echo "FAIL: missing input files:" >&2
    grep -oE "File \`[^']*' not found" main.log | sort -u >&2
    fail=1
fi
[ -f main.pdf ] || { echo "FAIL: no PDF produced" >&2; fail=1; }
[ "$fail" -eq 0 ] || exit 1

# --- package -------------------------------------------------------------
cp main.pdf "../factorybench-${VENUE}-workshop.pdf"
rm -f main.aux main.log main.blg main.out main.pdf   # keep main.bbl: portals need it
cd ..
rm -f "factorybench-${VENUE}-workshop-source.zip"
( cd "${VENUE}-source" && zip -q -r "../factorybench-${VENUE}-workshop-source.zip" . )

echo
echo "OK  dist/factorybench-${VENUE}-workshop.pdf"
echo "OK  dist/factorybench-${VENUE}-workshop-source.zip"
