FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV NAME World

CMD ["python", "app.py"]
