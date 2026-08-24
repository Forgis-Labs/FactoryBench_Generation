# Two builds share every line of content via _paper_body.tex:
#   make deanon  ->  main.pdf       (real author names + Forgis-Labs GitHub URL)
#   make anon    ->  main_anon.pdf  (blinded authors + anonymous.4open.science)
# Edit content in _paper_body.tex — never in main.tex or main_anon.tex,
# which are 5-line wrappers whose only job is to flip the \anon toggle.

LATEXMK ?= latexmk -pdf -interaction=nonstopmode

.PHONY: all deanon anon clean

all: deanon anon

deanon:
	$(LATEXMK) main.tex

anon:
	$(LATEXMK) main_anon.tex

clean:
	$(LATEXMK) -C
	-rm -f main_anon.*
