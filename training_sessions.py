import os
import keras
import numpy as np
import datetime
import re 

from data import outputs

def nat_sort(x): 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(x, key = alphanum_key)

class SessionGroup():
    def __init__(self, name) -> None:
        self.sessions_folder = os.path.join(os.path.dirname(__file__), "sessions", name)
        if not os.path.exists(self.sessions_folder):
            os.mkdir(self.sessions_folder)

    def latest(self):
        folders = os.listdir(self.sessions_folder)
        return os.path.join(self.sessions_folder, nat_sort(folders)[-1])

    def new_sess(self, path):
        return self.TrainingSession(path=path, group=self)
    
    def new_sess(self, models):
        return self.TrainingSession(models=models, group=self)

    class TrainingSession():
        def __init__(self, path, group) -> None:
            self.group = group
            models = dict()
            self.model_dir = os.path.join(path, "models")
            for model in os.listdir(self.model_dir):
                models.update({model : keras.models.load_model(os.path.join(self.model_dir, model))})
            self.models = models
            self.path = path
            self.out_dir = os.path.join(path, "out")
            self.history_dir = os.path.join(self.path, "history.npy")
            self.history = list(np.load(self.history_dir))

        def __init__(self, models, group) -> None:
            date = datetime.datetime.now()
            folder_name = f"{date.year}-{date.month}-{date.day}-{date.hour}-{date.minute}-{date.second}"
            self.path = os.path.join(group.sessions_folder, folder_name)
            os.mkdir(self.path)
            self.model_dir = os.path.join(self.path, "models") 
            os.mkdir(self.model_dir)
            for model in models:
                os.mkdir(os.path.join(self.model_dir, model))
            self.models = models
            self.out_dir = os.path.join(self.path, "out")
            os.mkdir(self.out_dir)
            self.history_dir = os.path.join(self.path, "history.npy")
            self.history = []
            self.group = group

        def save(self):
            for model in self.models:
                self.models[model].save(os.path.join(self.model_dir, model))
            open(self.history_dir).close()
            np.save(np.array(self.history))

        def save_samples(self, samples):
            for sample in samples:
                outputs.save_image(sample, self.out_dir)

        def save_plot(self, samples, n=4):
            outputs.save_plot(samples, n, self.out_dir)

