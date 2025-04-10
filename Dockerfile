# Use a minimal base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for image processing and Torch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies (ensuring CPU version of Torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .


# Remove unnecessary files to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* ~/.cache/pip

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--access-logfile", "-", "main:app"]
