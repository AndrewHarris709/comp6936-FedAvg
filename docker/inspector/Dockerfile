FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./generators ./generators
COPY ./linearRegression ./linearRegression
COPY ./dash_inspector.py .
COPY ./generation.config.json ./generation.config.json

CMD [ "python", "./dash_inspector.py", "http://fed-server:5000", "generation.config.json"]
