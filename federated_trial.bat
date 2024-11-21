start /b python flaskFedAvg-server.py -p flaskFederated.server.params.json

timeout /t 5 /nobreak > NUL

start /b python flaskFedAvg-client.py -p flaskFederated.client1.params.json
start /b python flaskFedAvg-client.py -p flaskFederated.client2.params.json
start /b python flaskFedAvg-client.py -p flaskFederated.client3.params.json
start /b python flaskFedAvg-client.py -p flaskFederated.client4.params.json
start /b python flaskFedAvg-client.py -p flaskFederated.client5.params.json
start /b python dash_inspector.py

timeout /t 5 /nobreak > NUL

for /l %%x in (1, 1, 5) do (
    timeout /t 2 /nobreak > NUL
    curl http://localhost:5000
    echo ""
    timeout /t 2 /nobreak > NUL
    curl http://localhost:5000/test
    echo ""
)

echo Iterations Done
timeout /t 5 /nobreak > NUL
