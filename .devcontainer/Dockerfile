# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Configure conda environment
RUN conda install python=3.9 pandas seaborn scipy scikit-learn tqdm ipykernel pytorch
RUN conda install autograd ipywidgets 
RUN conda install -c conda-forge numpyro arviz
RUN conda install -c conda-forge plotly
RUN conda install -c conda-forge numba
RUN pip install torchmetrics
RUN pip install pyro-ppl
RUN pip install pymc3

# Install RRMPG
RUN mkdir /tmp-pkg 
RUN git clone https://github.com/kratzert/RRMPG /tmp-pkg/RRMPG
WORKDIR /tmp-pkg/RRMPG
RUN python setup.py install
RUN rm -rf /tmp-pkg