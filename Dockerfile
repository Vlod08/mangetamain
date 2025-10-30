# Dockerfile for mangetamain Streamlit app
FROM python:3.11-slim

# Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /opt/mangetamain

# Copy project files
COPY pyproject.toml poetry.lock* /opt/mangetamain/

# Install dependencies first (for better caching)
RUN pip install --upgrade pip \
    && pip install "poetry==2.2.0" \
    && poetry install --no-interaction --no-root

# Copy source code
COPY . /opt/mangetamain/

# Install the project
RUN poetry install --no-interaction

EXPOSE 8501

# Run Streamlit app
CMD ["poetry", "run", "streamlit", "run", "src/mangetamain/app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# To build the Docker image, use: docker build -t mangetamain:latest .
# To run the Docker container, use: docker run --rm -p 8501:8501 mangetamain:latest
# The app can be accessed at http://localhost:8501 when the container is running.