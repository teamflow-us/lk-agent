FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multi-user-transcriber.py .

CMD ["python", "multi-user-transcriber.py", "start"]
