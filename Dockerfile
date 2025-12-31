FROM python:3.11-slim

WORKDIR /app

# Install dependencies
# Note: Build context should be set to lk-agent directory
COPY py_lk_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared /app/shared

# Copy agent code
COPY py_lk_agent /app/py_lk_agent

WORKDIR /app/py_lk_agent

CMD ["python", "ptt-transcriber.py", "start"]
