FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install required libraries
RUN pip install fastapi uvicorn pydantic openai openenv-core

EXPOSE 8080

# Run the server to handle the mandatory /reset ping
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
