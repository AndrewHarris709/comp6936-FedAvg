FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./federated ./federated
COPY ./generators ./generators
COPY ./linearRegression ./linearRegression
COPY ./Client.py ./Client.py
COPY ./flaskFederated.client.params.json .
COPY ./generation.config.json ./generation.config.json

CMD [ "python", "./Client.py", "-p", "flaskFederated.client.params.json" ]
