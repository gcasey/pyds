"""message.py
"""

from termcolor import colored


def info(msg):
    print colored("[info]: %s" % msg, 'green')


def warn(msg):
    print colored("[warn]: %s" % msg, 'magenta')

    
def fail(msg):
    print colored("[fail]: %s" % msg, 'red')