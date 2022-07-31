import subprocess
from threading import Thread

def run(argv:list = []):
    thread = Thread(target=starter)
    thread.start()

def starter(argv:list = []):
    subprocess.call(["python", 'visualizer\\network_visualizer.py'], shell=False)