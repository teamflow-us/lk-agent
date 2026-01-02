FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all agent code
COPY . /app

# Example environment variables for customization:
# - AGENT_TTS_VOICE: alloy, echo, fable, onyx, nova, shimmer (default: nova)
# - AGENT_TTS_MODEL: tts-1 or tts-1-hd (default: tts-1)
# - AGENT_TTS_SPEED: 0.25 to 4.0 (default: 1.0)
# - AGENT_LLM_MODEL: gpt-3.5-turbo, gpt-4-turbo, etc. (default: gpt-3.5-turbo)
# - AGENT_INSTRUCTIONS: Custom system instructions for the agent

# Default to running the conversational agent
CMD ["python", "multi-user-conversational-agent.py", "start"]
