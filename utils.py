
import os
from contextlib import contextmanager

third_dir = os.path.join(os.path.dirname(__file__), 'third')


@contextmanager
def temp_chdir(path):
    prev_path = os.getcwd()
    os.chdir(path)
    yield prev_path
    os.chdir(prev_path)
