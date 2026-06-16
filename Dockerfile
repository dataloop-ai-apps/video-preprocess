FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_opencv

WORKDIR /app

USER root

# ffmpeg is required as the backend for PyAV (libav* shared libraries).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
