FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./federated ./federated
COPY ./generators ./generators
COPY ./linearRegression ./linearRegression
COPY ./Server.py .
COPY ./flaskFederated.server.params.json .
COPY ./generation.config.json .

EXPOSE 5000

CMD [ "python", "./Server.py", "-p", "flaskFederated.server.params.json" ]
