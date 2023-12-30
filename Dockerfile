FROM rocm/tensorflow:rocm5.4-tf2.11-dev

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /opt/project
COPY requirements.txt /opt/project
COPY ./src /opt/project


RUN apt install graphviz -y
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /opt/project/requirements.txt

CMD ["python3", "runner.py"]
