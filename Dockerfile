FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y libgomp1

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# Pastikan model di path ini
COPY model /app/model  

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Cloud Run
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
