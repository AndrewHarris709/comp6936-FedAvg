FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./flaskFederated ./flaskFederated
COPY ./generators ./generators
COPY ./linearRegression ./linearRegression
COPY ./dash_inspector.py .

CMD [ "python", "./dash_inspector.py" ]
