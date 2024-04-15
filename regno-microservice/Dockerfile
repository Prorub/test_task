FROM python:3.9

RUN python -m pip install flask flask-cors gunicorn numpy pandas catboost

WORKDIR /ml_service

ADD app app
ADD logs logs
ADD ml ml
ADD requirements.txt requirements.txt

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "app.server:app" ]