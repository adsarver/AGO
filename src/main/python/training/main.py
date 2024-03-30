import os
from os.path import dirname, join as pjoin
path = os.getcwd()

directory = pjoin(path, 'dataset')

os.chdir(path)

files = ['scrape', 'tag', 'save', 'select']

if __name__ == 'main':
    if not os.path.isdir(directory):
        os.mkdir(directory)

[os.system('python3 ' + f'{file}.py dataset') for file in files]
