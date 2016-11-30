
MY_POSTER = my_poster

PAPER = ${MY_POSTER}

all: ${PAPER}.pdf

quick: ${PAPER}.tex
	pdflatex ${PAPER}

${PAPER}.pdf: clean ${PAPER}.tex quick
	pdflatex ${PAPER}
	bibtex ${PAPER}.aux
	pdflatex ${PAPER}
	pdflatex ${PAPER}

clean:
	rm -f ${PAPER}.aux ${PAPER}.blg ${PAPER}.pdf ${PAPER}.bbl ${PAPER}.log
