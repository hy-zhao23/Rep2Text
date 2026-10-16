import os
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

def set_fonts():
    calibri_dir = os.path.join(os.path.expanduser("~"), ".fonts/Calibri")
    for fname in os.listdir(calibri_dir) if os.path.isdir(calibri_dir) else []:
        if fname.lower().endswith(('.ttf', '.otf')):
            font_manager.fontManager.addfont(os.path.join(calibri_dir, fname))

    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.sans-serif'] = ['Calibri']

    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 8