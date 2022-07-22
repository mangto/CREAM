import os, sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def stop(message:str=""):
    if (message == ""):
        os.system("pause")
    else:
        os.system(f"echo {str(message)}&pause>nul")
    sys.exit()

def clear():
    os.system("cls")
    
def out(message, color, bold:bool=False,underline:bool=False):
    special = ''
    if (bold == True): special += bcolors.BOLD
    if (underline == True): special += bcolors.UNDERLINE

    print(f"{color}{special}{message}{bcolors.ENDC}")