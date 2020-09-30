FROM continuumio/miniconda3

COPY environment.yml .
COPY run.py .

RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/deepl_env/bin:$PATH
RUN /bin/bash -c "source activate deepl_env"

CMD [ "python", "run.py"]
