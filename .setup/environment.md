# Domino Compute Environment Setup
Make the below changes to the Domino compute environment for the workshop.
Leave all other entries as Default

### Name:
```
Fraud-Detection-Workshop
```

### Base Environment / Image:

Select:
```
Start from a custom base image
```
From:
```
quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.5-domino6.1-standard
```
### Description
```
6.1 Domino Standard Environment Py3.10 R4.5
Ubuntu 22.04
Python 3.10
R 4.5
Jupyter, Jupyterlab, VSCode, Rstudio
```

### Visibility
```
Globally Accessible
```

### Additional Docker File Instructions:
```
# Install uWSGI pre-requisites
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add missing library for uWSGI
ARG LDFLAGS=-fno-lto
ENV LDFLAGS=-fno-lto
ENV PYTHONPATH="${PYTHONPATH}:/mnt/code"

# Install Flask & uWSGI
RUN pip install --no-cache-dir Flask Flask-Compress Flask-Cors uwsgi six prometheus-client ydata_profiling streamlit st-pages streamlit-extras altair-saver "nbclient>=0.6.7,<0.7" "nbconvert>=6.5.4,<7" "lxml[html_clean]"
```

### Plugable Workspace Tools
```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  supportedFileExtensions: [ ".ipynb" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```
Build It
