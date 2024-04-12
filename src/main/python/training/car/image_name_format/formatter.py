from pathlib import Path
import pandas as pd
import os
from PIL import Image
import concurrent.futures
from pathlib import Path
from numpy.random import choice
from string import ascii_letters


## Change depending on input image set
DELIMITER = '$$' 
MAKE_IDX = 0
MODEL_IDX = 1
YEAR_IDX = 2

## Do not change
NDELIMITER = '_' 
MAX_WORKERS = 12
CHAR = list(ascii_letters)


def setAnno(file: Path):
    img = Image.open(Path(file))
    label = os.path.basename(file)
    if '$' in label:
        delim = str(label).split(DELIMITER)
        label = '_'.join(delim[:3])
    else:
        label.replace(' ', '_')
    return label, img


input_dir = os.path.join(os.getcwd(), 'images')
output_dir = os.path.join(Path(os.getcwd()).parents[0], 'pytorch', 'exterior')
labels = []
files = list(Path(input_dir).rglob("*.[Jj][Pp][Gg]"))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for file in files:
        futures.append(executor.submit(setAnno, file))
    for future in concurrent.futures.as_completed(futures):
        temp = future.result()
        temp[1].save(os.path.join(output_dir, temp[0] + '_{}.jpg'.format(''.join(choice(CHAR, 4)))))

		