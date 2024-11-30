from generators.data_gen import CholeskyGenerator
import json

def from_config(file: str):
    with open(file, 'r') as file:
        data = json.load(file)

    generator = CholeskyGenerator(corr=data['corr'], shifts=data['shifts'], mults=data['mults'])
    return generator
