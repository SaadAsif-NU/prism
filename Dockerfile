FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY prism ./prism

RUN pip install --no-cache-dir .

# Drop into the interactive SQL shell by default.
ENTRYPOINT ["prism"]
