# QCFD MPS solver: the 21-rod bundle solver, the report figures and the tutorial notebook.
#
#   docker build -t qcfd-mps .                                # CPU-only torch wheel
#   docker build -t qcfd-mps:cuda --build-arg TORCH_INDEX= .  # PyPI wheel, bundles CUDA
#   docker run --rm -p 8888:8888 qcfd-mps                     # JupyterLab on the tutorial: open the printed URL
#   docker run --rm -it -v "$PWD/runs:/work/data/results" qcfd-mps make check
#   docker run --rm -v "$PWD/fig:/work/report/fig" qcfd-mps make report
#   docker run --rm -it --gpus all qcfd-mps:cuda bash         # GPU runs (see README)
FROM python:3.13-slim

ARG TORCH_VERSION=2.15.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# LaTeX for the report figures (matplotlib usetex).
RUN apt-get update && apt-get install -y --no-install-recommends \
        make texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super dvipng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN if [ -n "${TORCH_INDEX}" ]; then \
        pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}; \
    else pip install --no-cache-dir torch==${TORCH_VERSION}; fi \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . \
    && python -c "import tensnet as tt, torch; print('tensnet', tt.__version__, '| torch', torch.__version__)"

ENV OMP_NUM_THREADS=4 MPLBACKEND=Agg
EXPOSE 8888
CMD ["make", "notebook"]
