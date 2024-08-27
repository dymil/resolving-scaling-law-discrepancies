# syntax=docker/dockerfile:1

FROM quay.io/jupyter/base-notebook
# not bind mounting b/c I might want to change things
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
pip install -r /tmp/requirements.txt
# conda pip woohoo!
USER root
RUN apt update && apt -y install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super git
USER jovyan