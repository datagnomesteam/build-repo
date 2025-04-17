FROM python:3.10-slim

WORKDIR /app

# Install Java for PySpark, dos2unix for line ending conversion, and other utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre \
    procps \
    curl \
    dos2unix \
    aria2 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Java home for PySpark - use a more portable way to find Java
RUN echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))" >> ~/.bashrc
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download and extract allCountries.txt directly to current directory
RUN aria2c --max-tries=50 --check-certificate=false https://datagnomesgroup.ddns.net/allCountries.zip && \
    unzip allCountries.zip && \
    rm allCountries.zip

# Convert line endings to Unix format (important for scripts when developed on Windows)
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Run both services with the entrypoint script
CMD ["/app/entrypoint.sh"] 