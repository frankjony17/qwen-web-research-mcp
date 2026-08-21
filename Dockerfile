FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
ENV OLLAMA_HOST_IP=host.docker.internal

EXPOSE 8000

CMD ["python", "-m", "web_research.server"]
