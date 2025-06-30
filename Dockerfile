# Use official Python image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variable to avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Set environment variable to buffer stdout/stderr (good for logging)
ENV PYTHONUNBUFFERED 1

# Run the application
CMD ["python", "app.py"]
