##########
## Builder stage: create a uv-managed virtualenv with all deps + project
##########
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Add uv binary to the image (matches approach from the uv docs)
COPY --from=ghcr.io/astral-sh/uv:0.5.24-debian-slim /uv /uvx /bin/

# Use a project-local virtualenv managed by uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

# First sync only dependencies for better layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Now add the rest of the source and install the project itself
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


##########
## Runtime stage: lightweight image with just the virtualenv + model
##########
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the virtualenv built by uv (includes the installed booking_sentiment package)
COPY --from=builder /app/.venv /app/.venv

# Copy only what we need at runtime: configs and the fine‑tuned model
COPY configs ./configs
COPY artifacts/finetuned_model ./artifacts/finetuned_model

# By default start in interactive inference mode, using the quick config.
# You can override the command, e.g.:
#   docker run --rm -it booking-sentiment-model evaluate --config configs/quick.json
ENTRYPOINT ["booking-sentiment"]
CMD ["inference", "--config", "configs/quick.json"]

