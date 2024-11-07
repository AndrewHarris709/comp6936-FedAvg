from time import sleep

class Client():
    def __init__(self, name):
        self.name = name

    def train(self, weights):
        sleep(5)
        print(f"{self.name} is here! Weights: {weights}")
        return {"weights": self.name, "numRecords": len(self.name)}