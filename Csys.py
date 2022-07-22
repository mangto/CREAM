import os

def Pause(message:str):
    os.system(f"echo {str(message)}&pause>nul")
