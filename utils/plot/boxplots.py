import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os


################################################
################ Barbox plots ##################
################################################

def plot_barbox(stats, title, ylabel, save_path):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [stats[k] for k in stats.keys()],
        tick_labels = list(stats.keys()),
        patch_artist=True,
        showmeans=True,
        meanline=True
    )

    ax.legend(['Mean'], loc='upper right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

