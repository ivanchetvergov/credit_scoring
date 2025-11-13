FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# устанавливаем PYTHONPATH, чтобы Python мог находить модули в src/ и scripts/
ENV PYTHONPATH="${PYTHONPATH}:/app"

# запускает обучение всех моделей без кросс-валидации для быстрого теста.
CMD ["python", "-u", "scripts/train_model.py", "--no-cv"]