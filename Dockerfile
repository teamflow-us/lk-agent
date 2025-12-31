FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ptt-transcriber.py .

CMD ["python", "ptt-transcriber.py", "start"]
