
# Use an official Python runtime as a parent image
FROM python:3.12.8

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements_fastapi.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_fastapi.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
