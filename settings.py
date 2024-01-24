import os 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')

PARAM_DIR = os.path.join(BASE_DIR, 'parameters')
OUT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'asset')

if not os.path.exists(PARAM_DIR):
    os.mkdir(PARAM_DIR)

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
    
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)