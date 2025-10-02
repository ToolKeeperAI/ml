FROM python:3.12-slim

WORKDIR /app

COPY app/ app/
COPY requirements.txt .

RUN apt-get update && apt-get install -y wget curl jq && rm -rf /var/lib/apt/lists/* \
    && curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/R4sdbN0ae_0PPA" \
    | jq -r '.href' \
    | xargs wget -O app/models/fasterrcnn.pth

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
