FROM python:3.10-slim

WORKDIR /app

# system dependencies (important for chroma + builds)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -r requirements.txt

# EXPOSE 8501

CMD ["python", "main.py"]