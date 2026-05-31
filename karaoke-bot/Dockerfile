FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata && \
    apt-get install -y ffmpeg libass-dev && \
    rm -f /opt/conda/bin/ffmpeg /opt/conda/bin/ffprobe || true && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-large-v3', device='cpu')"
RUN python -c "from demucs import pretrained; pretrained.get_model('htdemucs')"

COPY bot.py karaoke_utils.py ./

CMD ["python", "-u", "bot.py"]
