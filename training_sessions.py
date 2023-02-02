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

    def load_sess(self, path, relative=False):
        print(self.sessions_folder)
        if relative:
            path = os.path.join(str(self.sessions_folder), str(path))
        sess = self.TrainingSession()
        sess.group = self
        models = dict()
        sess.model_dir = os.path.join(path, "models")
        for model in os.listdir(sess.model_dir):
            models.update({model : keras.models.load_model(os.path.join(sess.model_dir, model))})
        sess.models = models
        sess.path = path
        sess.out_dir = os.path.join(path, "out")
        sess.history_dir = os.path.join(sess.path, "history.npy")
        sess.history = list(np.load(sess.history_dir))
        return sess
    
    def new_sess(self, models):
        sess = self.TrainingSession()
        date = datetime.datetime.now()
        folder_name = f"{date.year}-{date.month}-{date.day}-{date.hour}-{date.minute}-{date.second}"
        sess.path = os.path.join(self.sessions_folder, folder_name)
        os.mkdir(sess.path)
        sess.model_dir = os.path.join(sess.path, "models") 
        os.mkdir(sess.model_dir)
        for model in models:
            os.mkdir(os.path.join(sess.model_dir, model))
        sess.models = models
        sess.out_dir = os.path.join(sess.path, "out")
        os.mkdir(sess.out_dir)
        sess.history_dir = os.path.join(sess.path, "history.npy")
        sess.history = []
        sess.group = self
        return sess

    class TrainingSession():
        def __init__(self) -> None:
            pass

        def save(self):
            for model in self.models:
                self.models[model].save(os.path.join(self.model_dir, model))
            if os.path.exists(self.history_dir):
                open(self.history_dir).close()
            np.save(self.history_dir, np.array(self.history))

        def save_samples(self, samples):
            for sample in samples:
                outputs.save_image(sample, self.out_dir)

        def save_plot(self, samples, n=4):
            outputs.save_plot(samples, n, self.out_dir)

