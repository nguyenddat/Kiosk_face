FROM python:3.12.6

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
curl 

RUN apt-get install -y gcc g++ python3-dev

RUN mkdir /app
WORKDIR /app