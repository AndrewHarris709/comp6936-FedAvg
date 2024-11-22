FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./flaskFederated ./flaskFederated
COPY ./generators ./generators
COPY ./linearRegression ./linearRegression
COPY ./flaskFedAvg-client.py .
COPY ./flaskFederated.client.params.json .

CMD [ "python", "./flaskFedAvg-client.py", "-p", "flaskFederated.client.params.json" ]
