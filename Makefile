.PHONY: check smoke report notebook warmstart

check:      ## unit test, solver self-checks, probe-vs-dense forces, dense twin (a few minutes)
	scripts/check

smoke:      ## 256^2 bundle: warm start, 3 RK4 steps, per-rod forces (a few minutes)
	scripts/run smoke
	scripts/post smoke

report:     ## regenerate the report figures and tables in report/fig from data/archive
	cd report && python3 plots.py && python3 cost_scaling.py

notebook:   ## JupyterLab on the tutorial; open the printed URL (port 8888)
	jupyter lab --notebook-dir=tutorial --ip=0.0.0.0 --no-browser --allow-root

warmstart:  ## rebuild data/initial_conditions from data/star_ccm/star_t2500.vtk
	cd scripts && for p in 5 6; do python3 bundle_simulation.py warmstart --vtk ../data/star_ccm/star_t2500.vtk --p $$p --sizes 5; done
