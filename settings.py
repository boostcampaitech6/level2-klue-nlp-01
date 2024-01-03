import os 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
DEV_DIR = os.path.join(DATA_DIR, 'dev')
TEST_DIR = os.path.join(DATA_DIR, 'test')

PARAM_DIR = os.path.join(BASE_DIR, 'parameters')
OUT_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(PARAM_DIR):
    os.mkdir(PARAM_DIR)

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
