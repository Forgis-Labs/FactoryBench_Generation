# Every PDF, one source of truth.
#
#   make neurips              ->  neurips_tex/main.pdf       (NeurIPS, real author names)
#   make anon                 ->  neurips_tex/main_anon.pdf  (NeurIPS, double-blind)
#   make iclr                 ->  iclr_tex/main_iclr.pdf     (ICLR 2027, double-blind)
#   make workshops            ->  every workshop PDF in WORKSHOPS below
#   make workshop-wmphysai    ->  workshop_tex/main_wmphysai.pdf (one per venue)
#   make all                  ->  the three full-paper builds
#   make everything           ->  all + workshops
#   make archives             ->  dist/factorybench-<venue>-workshop.{pdf,zip}
#   make clean                ->  drop LaTeX aux files (keeps the PDFs)
#
# FULL PAPER. Every line of prose lives in paper/_content.tex and is shared
# verbatim by the three full-paper builds. The files under neurips_tex/ and
# iclr_tex/ are wrappers whose only job is to load the venue style package and
# flip toggles:
#
#   \ifanon              blinded author block          (anon + iclr: true)
#   \ifneuripschecklist  include the NeurIPS checklist (neurips only: true)
#   \paperbibstyle       bibliography style            (iclr: its own bst)
#
# WORKSHOPS. FactoryBench goes to several non-archival workshops, so those
# builds are structured the same way one level down. Each venue is a pair:
#
#   paper/_workshop_<venue>.tex     venue framing: title block, abstract,
#                                   introduction, two connector sentences
#   workshop_tex/main_<venue>.tex   wrapper: style package, \workshoptitle,
#                                   page-limit switch
#
# and both pull the SAME body from paper/_workshop_core.tex (related work,
# framework, generation, evaluation, limitations, conclusion). Fix a result
# once in the core and every workshop PDF is correct on the next compile.
# A fourth toggle applies only to these:
#
#   \ifworkshopcompact   drop 3 optional floats, ~6pp body -> ~4pp (6-page venues)
#
# To add a venue, copy paper/_workshop_TEMPLATE.tex and
# workshop_tex/main_TEMPLATE.tex, then append the name to WORKSHOPS below.
#
# Edit the shared sources and every PDF changes. Never edit a wrapper to change
# content, and never copy content between venue directories.
#
# Note: all three bibliographies were consolidated into paper/references.bib.
# The per-venue copies had drifted apart, which is why the NeurIPS build used
# to emit undefined citations the ICLR build did not.

LATEX  ?= pdflatex -interaction=nonstopmode
BIBTEX ?= bibtex

# Workshop venues. One name per submission; each needs the two files above.
WORKSHOPS ?= wmphysai physunderstanding robotlearning

WORKSHOP_TARGETS  := $(addprefix workshop-,$(WORKSHOPS))
WORKSHOP_ARCHIVES := $(addprefix archive-,$(WORKSHOPS))

.PHONY: all everything neurips anon iclr workshops archives clean \
        $(WORKSHOP_TARGETS) $(WORKSHOP_ARCHIVES)

all: neurips anon iclr

everything: all workshops

neurips:
	cd neurips_tex && $(LATEX) main.tex && $(BIBTEX) main && $(LATEX) main.tex && $(LATEX) main.tex

anon:
	cd neurips_tex && $(LATEX) main_anon.tex && $(BIBTEX) main_anon && $(LATEX) main_anon.tex && $(LATEX) main_anon.tex

iclr:
	cd iclr_tex && $(LATEX) main_iclr.tex && $(BIBTEX) main_iclr && $(LATEX) main_iclr.tex && $(LATEX) main_iclr.tex

workshops: $(WORKSHOP_TARGETS)

# One rule builds every venue: the stem is the venue name.
workshop-%:
	cd workshop_tex && $(LATEX) main_$*.tex && $(BIBTEX) main_$* && $(LATEX) main_$*.tex && $(LATEX) main_$*.tex

# Self-contained submission archives, one per venue. Portals want a zip that
# compiles on their machine, so each carries its own copy of every input it
# reads: wrapper, shared sources, style file, bibliography, and only the
# figures that build actually pulls in. See make_archive.sh.
archives: $(WORKSHOP_ARCHIVES)

archive-%: workshop-%
	./make_archive.sh $*

clean:
	-rm -f neurips_tex/main.aux neurips_tex/main.bbl neurips_tex/main.blg neurips_tex/main.log neurips_tex/main.out
	-rm -f neurips_tex/main_anon.aux neurips_tex/main_anon.bbl neurips_tex/main_anon.blg neurips_tex/main_anon.log neurips_tex/main_anon.out
	-rm -f iclr_tex/main_iclr.aux iclr_tex/main_iclr.bbl iclr_tex/main_iclr.blg iclr_tex/main_iclr.log iclr_tex/main_iclr.out
	-rm -f workshop_tex/*.aux workshop_tex/*.bbl workshop_tex/*.blg workshop_tex/*.log workshop_tex/*.out
