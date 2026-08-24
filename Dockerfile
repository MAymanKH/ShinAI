FROM python:3.13-slim-bookworm

# Upgrade system packages to patch vulnerabilities and install dependencies
# (e.g., tgcrypto, chromadb, sentence-transformers)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Dependency metadata first so the install layer caches independently of source.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY shin_ai/__init__.py ./shin_ai/

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run on container start
CMD ["python", "main.py"]
