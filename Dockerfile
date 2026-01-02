FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all agent code
COPY . /app

# Default to running the conversational agent
CMD ["python", "multi-user-conversational-agent.py"]
