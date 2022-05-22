FROM python:alpine3.8

COPY . /app/

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]
