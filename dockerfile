# Use the official slim Python 3.9 image
FROM python:3.9-slim

# Install git, git-lfs and their dependencies
RUN apt-get update \
    && apt-get install -y git git-lfs curl ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository into /app and fetch all LFS objects
RUN git clone https://github.com/xartex24/pydaling-paris.git /app \
    && cd /app \
    && git lfs pull

# Set the working directory
WORKDIR /app

# Copy only the requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optionally remove the .git folder to shrink the image
RUN rm -rf .git

# Copy the rest of the application code
COPY . .

# Expose port 8080 for Streamlit
ENV PORT 8080
EXPOSE 8080

# Launch the Streamlit app on container start
CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
