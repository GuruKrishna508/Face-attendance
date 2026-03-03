FROM python:3.11-slim

WORKDIR /app

# Install pre-compiled dlib dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dlib from a pre-built wheel (no compilation)
RUN pip install --upgrade pip
RUN pip install dlib==19.24.1 --extra-index-url https://pypi.org/simple/

# Install the rest
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT
