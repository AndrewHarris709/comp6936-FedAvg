from Client import Client
from Server import Server

a = Client("Ali")
b = Client("Andrew")
c = Client("Mahdi")
d = Client("James")
e = Client("Victoria")

server = Server(0.8, -1, [a, b, c, d, e])

server.start_clients()