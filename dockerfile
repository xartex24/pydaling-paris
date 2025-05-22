# Use the official slim Python 3.9 image
FROM python:3.9-slim

# Install git, git-lfs and their dependencies
RUN apt-get update \
    && apt-get install -y git git-lfs curl ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Clone your repo (pointer+LFS)
RUN git clone https://github.com/xartex24/pydaling-paris.git /app \
    && cd /app \
    && git lfs pull

# Set working dir
WORKDIR /app

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# (Опционално) изчистете .git за по-малък образ
RUN rm -rf .git

# Expose port and run
ENV PORT 8080
EXPOSE 8080

CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
