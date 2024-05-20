FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl python3.8 python3.8-distutils && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

RUN pip install --no-cache-dir torch torchvision torchaudio
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV PROJECT_ROOT /app

RUN mkdir $PROJECT_ROOT

WORKDIR $PROJECT_ROOT
COPY download.py download.py
RUN python download.py

COPY . $PROJECT_ROOT

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



