#!/bin/bash
killjobs() {
    for x in $(jobs | awk -F '[][]' '{print $2}' ) ; do 
        kill %$x
    done
}
trap killjobs EXIT

python Server.py -p linux.server.params.json &

sleep 2

for i in {1..5}
do
    python Client.py -p linux.client.params.json &
done

python dash_inspector.py http://localhost:5000
