FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev extras, no editable install yet)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source package and install it
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Copy app and supporting files
COPY app.py ./
COPY data/ ./data/
COPY configs/ ./configs/

EXPOSE 8050

CMD ["uv", "run", "python", "app.py"]