FROM python:3.13-slim-bookworm

# ffmpeg is required by faster-whisper for audio decoding; build-essential
# covers source builds for tgcrypto and chromadb's native extensions.
# Pinning to the base image's package set keeps rebuilds of the same commit
# reproducible -- rebase onto a newer python:3.13-slim tag to pick up
# security updates rather than running apt-get upgrade at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first so the install layer caches independently of source.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY shin_ai/__init__.py ./shin_ai/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The container mounts config.yaml, session files and API keys. Run as an
# unprivileged user so a compromise inside the bot cannot write to them or
# to anything else in the image.
RUN useradd --create-home --uid 10001 shinai \
    && chown -R shinai:shinai /app
USER shinai

# The bot has no listening socket, so liveness is "the process is still
# running and the interpreter is responsive".
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "main.py"]
