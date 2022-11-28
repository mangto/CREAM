import os, sys
from cream.tool.colors import *

def division(length:int, Return=False):
    if (length < 4): raise ValueError("Don't you think that's too short?")
    if Return: return "+"+"-"*(length-2)+"+"
    else: print("+"+"-"*(length-2)+"+")

def stop(message:str=""):
    if (message == ""): os.system("pause")
    else: os.system(f"echo {str(message)}&pause>nul")
    sys.exit()

def error(message:str="", name:str="Unknown"):
    out(f"Error Occured ({name}): " + str(message), FAIL, True)
    os.system("pause")
    sys.exit()

def clear():
    os.system("cls")
    
def out(message, color, bold:bool=False,underline:bool=False):
    special = ''
    if (bold == True): special += BOLD
    if (underline == True): special += UNDERLINE

    print(f"{color}{special}{message}{ENDC}")