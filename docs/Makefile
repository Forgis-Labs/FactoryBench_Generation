# Three PDFs, one source of truth.
#
#   make neurips  ->  neurips_tex/main.pdf       (NeurIPS, real author names)
#   make anon     ->  neurips_tex/main_anon.pdf  (NeurIPS, double-blind)
#   make iclr     ->  iclr_tex/main_iclr.pdf     (ICLR 2027, double-blind)
#   make all      ->  all three
#
# Every line of prose lives in paper/_content.tex and is shared verbatim by all
# three builds. The files under neurips_tex/ and iclr_tex/ are wrappers whose
# only job is to load the venue style package and flip toggles:
#
#   \ifanon              blinded author block          (anon + iclr: true)
#   \ifneuripschecklist  include the NeurIPS checklist (neurips only: true)
#   \paperbibstyle       bibliography style            (iclr: its own bst)
#
# Edit paper/_content.tex and every PDF changes. Never edit a wrapper to change
# content, and never copy content between venue directories.

LATEX  ?= pdflatex -interaction=nonstopmode
BIBTEX ?= bibtex

.PHONY: all neurips anon iclr clean

all: neurips anon iclr

neurips:
	cd neurips_tex && $(LATEX) main.tex && $(BIBTEX) main && $(LATEX) main.tex && $(LATEX) main.tex

anon:
	cd neurips_tex && $(LATEX) main_anon.tex && $(BIBTEX) main_anon && $(LATEX) main_anon.tex && $(LATEX) main_anon.tex

iclr:
	cd iclr_tex && $(LATEX) main_iclr.tex && $(BIBTEX) main_iclr && $(LATEX) main_iclr.tex && $(LATEX) main_iclr.tex

clean:
	-rm -f neurips_tex/main.aux neurips_tex/main.bbl neurips_tex/main.blg neurips_tex/main.log neurips_tex/main.out
	-rm -f neurips_tex/main_anon.aux neurips_tex/main_anon.bbl neurips_tex/main_anon.blg neurips_tex/main_anon.log neurips_tex/main_anon.out
	-rm -f iclr_tex/main_iclr.aux iclr_tex/main_iclr.bbl iclr_tex/main_iclr.blg iclr_tex/main_iclr.log iclr_tex/main_iclr.out
