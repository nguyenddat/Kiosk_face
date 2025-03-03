FROM python:3.12.6

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
curl 

RUN apt-get update && apt-get install -y \
    curl \
    gcc g++ python3-dev \
    libgl1-mesa-glx libglib2.0-0 

    RUN mkdir /kiosk_face
WORKDIR /kiosk_face
COPY . /kiosk_face/
RUN pip install -r requirements.txt

# RUN python app/main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]
