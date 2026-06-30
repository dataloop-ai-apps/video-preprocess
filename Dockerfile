FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_opencv

WORKDIR /app

USER root

# ffmpeg supplies the `ffprobe` binary used via subprocess in main.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    --index-url https://artifacts.dell.com/artifactory/api/pypi/python/simple \
    --trusted-host artifacts.dell.com \
    -r /app/requirements.txt

USER 1000
