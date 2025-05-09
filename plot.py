import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def plot_visualization(*arrays, save_filename: str, num_items: int):
    length = len(arrays)


    # Plot original and reconstructed images
    fig, axes = plt.subplots(length, num_items, figsize=(10, 2))
    for i in range(num_items):

        for j in range(length):
            axes[j, i].imshow(
                make_grid(arrays[j][i].unsqueeze(0), normalize=True).numpy()
            )
            axes[j, i].axis("off")
    plt.tight_layout()
    plt.savefig(save_filename)
