# Five PDFs. Three share one source of truth; the workshop builds share a
# second, each capped at its own venue's page limit.
#
#   make neurips   ->  neurips_tex/main.pdf         (NeurIPS, real author names)
#   make anon      ->  neurips_tex/main_anon.pdf    (NeurIPS, double-blind)
#   make iclr      ->  iclr_tex/main_iclr.pdf       (ICLR 2027, double-blind)
#   make wmphysai  ->  wmphysai_tex/main_wmphysai.pdf (World Models in Physical
#                       AI workshop @ NeurIPS 2026, double-blind, <=8 pages
#                       excluding references)
#   make robotlearning -> robotlearning_tex/main_robotlearning.pdf (8th Robot
#                       Learning Workshop @ NeurIPS 2026, <=6 pages, compact
#                       mode)
#   make physunderstanding -> physunderstanding_tex/main_physunderstanding.pdf
#                       (Physical Understanding for Decision-Making @ NeurIPS
#                       2026, <=9 main pages, appendix exempt so it ships the
#                       full shared appendix)
#   make all       ->  all six
#
# Every line of prose in the first three lives in paper/_content.tex and is
# shared verbatim. The files under neurips_tex/ and iclr_tex/ are wrappers
# whose only job is to load the venue style package and flip toggles:
#
#   \ifanon              blinded author block          (anon + iclr: true)
#   \ifneuripschecklist  include the NeurIPS checklist (neurips only: true)
#   \paperbibstyle       bibliography style            (iclr: its own bst)
#
# Edit paper/_content.tex and those three PDFs change together. Never edit a
# wrapper to change content, and never copy content between venue directories.
#
# The workshop builds share a second source of truth, appendix-free and
# trimmed to workshop length:
#
#   paper/_workshop_core.tex     method/results/limitations/conclusion,
#                                identical across every workshop build
#   paper/_workshop_header.tex   title block, links, teaser figure
#   paper/_workshop_<venue>.tex  that venue's abstract + intro, plus the two
#                                connector sentences the core expands
#
# Edit _workshop_core.tex once and every workshop PDF changes. Venues with a
# tighter limit set \workshopcompacttrue in their wrapper, which drops the
# core's optional floats (~2 pages) without losing any claim or number.
#
# To add a workshop: copy a paper/_workshop_<venue>.tex, copy a wrapper dir,
# add a target below. Never copy _workshop_core.tex.

LATEX  ?= pdflatex -interaction=nonstopmode
BIBTEX ?= bibtex

.PHONY: all neurips anon iclr wmphysai robotlearning physunderstanding clean

all: neurips anon iclr wmphysai robotlearning physunderstanding

neurips:
	cd neurips_tex && $(LATEX) main.tex && $(BIBTEX) main && $(LATEX) main.tex && $(LATEX) main.tex

anon:
	cd neurips_tex && $(LATEX) main_anon.tex && $(BIBTEX) main_anon && $(LATEX) main_anon.tex && $(LATEX) main_anon.tex

iclr:
	cd iclr_tex && $(LATEX) main_iclr.tex && $(BIBTEX) main_iclr && $(LATEX) main_iclr.tex && $(LATEX) main_iclr.tex

wmphysai:
	cd wmphysai_tex && $(LATEX) main_wmphysai.tex && $(BIBTEX) main_wmphysai && $(LATEX) main_wmphysai.tex && $(LATEX) main_wmphysai.tex

robotlearning:
	cd robotlearning_tex && $(LATEX) main_robotlearning.tex && $(BIBTEX) main_robotlearning && $(LATEX) main_robotlearning.tex && $(LATEX) main_robotlearning.tex

physunderstanding:
	cd physunderstanding_tex && $(LATEX) main_physunderstanding.tex && $(BIBTEX) main_physunderstanding && $(LATEX) main_physunderstanding.tex && $(LATEX) main_physunderstanding.tex

clean:
	-rm -f neurips_tex/main.aux neurips_tex/main.bbl neurips_tex/main.blg neurips_tex/main.log neurips_tex/main.out
	-rm -f neurips_tex/main_anon.aux neurips_tex/main_anon.bbl neurips_tex/main_anon.blg neurips_tex/main_anon.log neurips_tex/main_anon.out
	-rm -f iclr_tex/main_iclr.aux iclr_tex/main_iclr.bbl iclr_tex/main_iclr.blg iclr_tex/main_iclr.log iclr_tex/main_iclr.out
	-rm -f wmphysai_tex/main_wmphysai.aux wmphysai_tex/main_wmphysai.bbl wmphysai_tex/main_wmphysai.blg wmphysai_tex/main_wmphysai.log wmphysai_tex/main_wmphysai.out
	-rm -f robotlearning_tex/main_robotlearning.aux robotlearning_tex/main_robotlearning.bbl robotlearning_tex/main_robotlearning.blg robotlearning_tex/main_robotlearning.log robotlearning_tex/main_robotlearning.out
	-rm -f physunderstanding_tex/main_physunderstanding.aux physunderstanding_tex/main_physunderstanding.bbl physunderstanding_tex/main_physunderstanding.blg physunderstanding_tex/main_physunderstanding.log physunderstanding_tex/main_physunderstanding.out
