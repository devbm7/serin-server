
# Use an official Python runtime as a parent image
FROM python:3.12.8

# Install nginx for reverse proxy
RUN apt-get update && apt-get install -y nginx

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements_fastapi.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_fastapi.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Create nginx configuration
RUN echo 'server { \
    listen 80; \
    server_name _; \
    location / { \
        proxy_pass http://127.0.0.1:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_http_version 1.1; \
        proxy_set_header Upgrade $http_upgrade; \
        proxy_set_header Connection "upgrade"; \
    } \
}' > /etc/nginx/sites-available/default

# Expose port 80 for nginx
EXPOSE 80

# Create startup script
RUN echo '#!/bin/bash \n\
service nginx start \n\
uvicorn main:app --host 127.0.0.1 --port 8000 & \n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
