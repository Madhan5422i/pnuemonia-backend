# Use official Python image as base
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
