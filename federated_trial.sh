#!/bin/bash
killjobs() {
    for x in $(jobs | awk -F '[][]' '{print $2}' ) ; do 
        kill %$x
    done
}
trap killjobs EXIT

python flaskFedAvg-server.py -p flaskFederated.server.params.json &

sleep 5

python flaskFedAvg-client.py -p flaskFederated.client1.params.json &
python flaskFedAvg-client.py -p flaskFederated.client2.params.json &
python flaskFedAvg-client.py -p flaskFederated.client3.params.json &
python flaskFedAvg-client.py -p flaskFederated.client4.params.json &
python flaskFedAvg-client.py -p flaskFederated.client5.params.json &
python dash_inspector.py &

sleep 5

counter=1
while [ $counter -le 50 ]
do
    sleep 2
    curl http://localhost:5000
    echo ""
    sleep 2
    curl http://localhost:5000/test
    echo ""
    ((counter++))
done

echo Iterations Done
sleep 5
