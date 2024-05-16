FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

#FROM tiangolo/uvicorn-gunicorn-machine-learning:cuda9.1-python3.7
#FROM NVIDIA/nvidia-docker

RUN apt-get update && \
    apt-get install -y curl python3.8 python3.8-distutils && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*


RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3


ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install -r requirements.txt

# Копирование и запуск скрипта download.py
COPY download.py $PROJECT_ROOT
RUN python download.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


