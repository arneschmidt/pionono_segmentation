#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/arne/Documents/Research/Submissions/Pionono/images/examples/intra-observer_variability/"
# path = "/home/arne/Documents/Research/Submissions/Pionono/images/examples/inter-observer_variability/"
# path = "/home/arne/Documents/Research/Submissions/Pionono/images/examples/uncertainty"
dirs = os.listdir( path )
print(dirs)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f+ '.png', 'PNG')

resize()