import os
from os.path import dirname, join as pjoin
path = os.getcwd()

os.chdir(path)

directory = pjoin(path, 'dataset')

files = ['scrape', 'tag', 'save', 'select']

if __name__ == 'main':
    if not os.path.isdir(path):
        os.mkdir(path)

[os.system('python3 ' + f'{file}.py ' + directory) for file in files]
