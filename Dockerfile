FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn -b 0.0.0.0:$PORT --timeout 180 --workers 1 --threads 2 server:app
