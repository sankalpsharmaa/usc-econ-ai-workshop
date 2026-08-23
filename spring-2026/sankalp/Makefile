TEX = main.tex
PDF = main.pdf

all: $(PDF)

$(PDF): $(TEX)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(TEX)

watch:
	latexmk -xelatex -pvc -interaction=nonstopmode $(TEX)

clean:
	latexmk -C
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk

open: $(PDF)
	open $(PDF)

.PHONY: all watch clean open
