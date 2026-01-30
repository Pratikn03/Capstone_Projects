FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ src/
COPY services/ services/
COPY configs/ configs/

# Set python path
ENV PYTHONPATH=/app/src

# Run Streamlit
CMD ["streamlit", "run", "services/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
