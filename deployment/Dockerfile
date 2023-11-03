FROM python:3.9.18-slim

RUN pip install pipenv


WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "encoder.bin","./"]

EXPOSE 2912

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:2912", "predict:app"]