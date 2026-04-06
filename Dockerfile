FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the inference script
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]