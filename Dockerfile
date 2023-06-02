FROM python:3.10-slim

ARG work_dir="/work"

WORKDIR ${work_dir}

RUN apt-get update && apt-get install -y unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake g++ \
    curl

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="1.4.0"

RUN pip install --upgrade pip && pip uninstall -y virtualenv

RUN curl -sSSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml* ./poetry.lock* ./

RUN poetry install

ENV PYTHONPATH ${work_dir}:${PYTHONPATH}

CMD [ "jupyter-lab", "--ip", "0.0.0.0", "--allow-root", "--NotebookApp.token=''" ]
