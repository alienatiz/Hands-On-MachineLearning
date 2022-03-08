import os
import sys

assert sys.version_info >= (3, 5)
import sklearn

assert sklearn.__version__ >= "0.20"
import matplotlib
import matplotlib.pyplot as plt

# Setup matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ensemble"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
