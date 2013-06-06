"""message.py

Simple command-line messaging (with color).
"""


__license__ = "Apache License, Version 2.0"
__author__  = "Roland Kwitt, Kitware Inc., 2013"
__email__   = "E-Mail: roland.kwitt@kitware.com"
__status__  = "Development"


from termcolor import colored


def info(msg):
    print colored("[info]: %s" % msg, 'green')


def warn(msg):
    print colored("[warn]: %s" % msg, 'magenta')

    
def fail(msg):
    print colored("[fail]: %s" % msg, 'red')