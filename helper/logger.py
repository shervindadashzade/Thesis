import datetime
import os


class Logger:
    def __init__(self, path, append=False):
        self.path = path
        if not append:
            if os.path.exists(path):
                os.remove(path)

    def log(self, message):
        message = f'{datetime.datetime.now()}:{message}\n'
        with open(self.path,'a') as f:
            f.write(message)