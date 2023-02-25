FROM  python:3.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/UTC
RUN apt install --no-install-recommends -y tk-dev apt-utils tzdata locales

ENV LANG=C.UTF-8
RUN locale-gen en_US.UTF-8

# set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV TZ=Asia/Ho_Chi_Minh

RUN apt-get update && apt-get install -y --no-install-recommends
RUN apt update

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
RUN python3.9 -m pip install pip setuptools wheel
COPY requirements.txt .
RUN python3.9 -m pip install -r requirements.txt

CMD ["/bin/bash"]
