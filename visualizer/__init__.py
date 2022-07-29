import subprocess
from threading import Thread

def run():
    thread = Thread(target=starter)
    thread.start()

def starter():
    subprocess.call(["python",f'visualizer\\network_visualizer.py'], shell=False)