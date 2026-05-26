# Compile the EE364B poster
# Usage: make            — build poster.pdf
#        make clean      — remove build artifacts

MAIN := poster

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex   # second pass for cross-refs

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb $(MAIN).pdf
