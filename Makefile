scripts := $(wildcard experiments/simulations/*.py)

.PHONY: main.pdf all clean installdeps

all: main.pdf

main.pdf: main.tex figs
	latexmk -f -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $<

figures/graphs/%.pdf: figure_scripts/%.dot
	mkdir -p figures/graphs
	./figure_scripts/draw_graph.sh $< > $@

figures/ukbiobank/smoking_prediction/ukb_smoking_prediction.pdf: experiments/ukbiobank/smoking_prediction.py
	python3 $<

%.d: %.py
	@echo 'figures/simulations/$(@F:.d=)/%: $<\n\tpython3 $<' > $@

main.d: main.tex
	@sed -n -e '/^.*includegraphics[^{]*{\([^}]*\)}.*/{s//\1/;H};' \
	-e '$${x;s/\n/ /g;s/\(.\+\)/figs: \1/p}' main.tex > main.d

installdeps:
	sudo apt-get install graphviz
	sudo apt-get install texlive-full
	pip install -r experiments/simulations/requirements.txt

clean-tex:
	latexmk -CA
	rm -f diff-with-original-submission.tex
	rm -f gigascience/revision/reply.pdf

clean:
	latexmk -CA
	rm -f diff-with-original-submission.tex
	rm -f gigascience/revision/reply.pdf
	find -type f -name "*.d" -delete
	rm -rf figures/graphs
	rm -rf figures/simulations

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), installdeps)
ifneq ($(MAKECMDGOALS), download)
-include $(scripts:.py=.d)
-include main.d
endif
endif
endif
