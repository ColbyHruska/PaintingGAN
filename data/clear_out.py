import os

data_path = os.path.join(os.path.dirname(__file__),"out/")

for file in os.listdir(data_path):
    if(data_path.endswith(".png")):
        os.remove(os.path.join(data_path, file))