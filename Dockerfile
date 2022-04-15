FROM ubuntu:18.04
WORKDIR /usr/src/app
COPY . .
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install  python3-dev python3-pip 
RUN echo alias python=python3 >> ~/.bashrc
RUN echo alias pip=pip3 >> ~/.bashrc
RUN pip install tensorflow
RUN pip install keras
RUN pip install pygame

#CMD [ "python", "./run_DRL.py" ]