# One-shot reproduction targets for the EE364B final project.
#
# Usage:
#   make install           # editable install + dev deps
#   make test              # unit tests
#   make methods           # headline 6-way methods comparison
#   make identifiability   # active-set identifiability study
#   make diurnal           # stratified vs static on diurnal data
#   make pjm               # PJM-like 8-fuel economic dispatch
#   make mef               # FD-vs-autograd Jacobian + MEF accuracy
#   make sweep             # T and sigma sweeps
#   make experiments       # all of the above (long)
#   make paper             # compile paper/main.pdf
#   make all               # tests + experiments + paper

PYTHON ?= /opt/miniconda3/envs/cvxpy-docs/bin/python
PYFLAGS := -W ignore

.PHONY: install test methods identifiability diurnal pjm mef sweep \
        experiments figs paper all clean

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) $(PYFLAGS) -m pytest tests -q

methods:
	$(PYTHON) $(PYFLAGS) scripts/run_methods_comparison.py --seeds 0,1,2,3 --steps 600

identifiability:
	$(PYTHON) $(PYFLAGS) scripts/run_identifiability.py --seeds 0,1,2,3 --steps 500

diurnal:
	$(PYTHON) $(PYFLAGS) scripts/run_diurnal.py --seeds 0,1,2 --steps 500

pjm:
	$(PYTHON) $(PYFLAGS) scripts/run_pjm_like.py --seeds 0,1 --n_train 400 --steps 500

mef:
	$(PYTHON) $(PYFLAGS) scripts/run_mef_validation.py --seeds 0,1,2 --n_points 8 --steps 400

sweep:
	$(PYTHON) $(PYFLAGS) scripts/run_sweep.py --mode size --sizes 50,100,200,400 --seeds 0,1,2 --steps 400
	$(PYTHON) $(PYFLAGS) scripts/run_sweep.py --mode noise --noises 0.1,0.3,0.7,1.5 --seeds 0,1,2 --steps 400

experiments: methods identifiability diurnal pjm mef sweep
	@echo "All experiments done. Now run 'make figs' to render IEEE figures."

figs:
	$(PYTHON) $(PYFLAGS) scripts/make_paper_figures.py

paper: figs
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

all: test experiments paper

clean:
	rm -rf outputs/methods_comparison outputs/identifiability outputs/diurnal \
		outputs/pjm_like outputs/mef_validation outputs/sweep_size outputs/sweep_noise
	rm -f paper/main.{aux,log,out,toc,bbl,blg,pdf}

