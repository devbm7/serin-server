# Use an official Python runtime as a parent image
FROM python:3.12.8

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for pyaudio and health checks
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    curl \
    ffmpeg \
    gunicorn \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt . 

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --upgrade

RUN python -c "import kokoro; pipeline = kokoro.KPipeline(lang_code='a')"
RUN python -c "from rfdetr import RFDETRNano; model = RFDETRNano()"

# Copy the specified Python files into the container at /app
COPY config.py .
COPY supabase_config.py .
COPY fastapi_pipeline.py .
COPY start_fastapi_server.py .
COPY pipeline_config.yaml .
COPY evalm.py .
COPY pdf5.py .
COPY redisc.py .

# Copy local model directories if they exist (only for ASR)
# COPY whisper-medium/ ./whisper-medium

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run start_fastapi_server.py when the container launches
# CMD ["python", "start_fastapi_server.py"]
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "fastapi_pipeline:app"]
