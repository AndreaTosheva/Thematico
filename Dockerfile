FROM

# Environment variables
ENV POETRY_VERSION=1.8.2\
    POETRY_NO_INTERACTION=1\
    POETRY_VIRTUALENVS_CREATE=false\
    PYTHNONUNBUFFERED=1


# Install dependencies
RUN apt-get update && apt-get install -y\
    curl build-essential git && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set the working directory
WORKDIR /app

# Copy dependency files and install packages
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--"]