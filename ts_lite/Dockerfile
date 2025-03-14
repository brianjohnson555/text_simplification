# Use Python base image
FROM python:3.12

# Set working directory
WORKDIR /ts_lite

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]