import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os


def calculate_metrics(img1_path, img2_path):
    """
    Calculates MSE, PSNR, and SSIM between two images.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.

    Returns:
        tuple: MSE, PSNR, and SSIM values.
    """
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))

    mse = np.mean((img1 - img2) ** 2)
    psnr_value = 20 * np.log10(255 / np.sqrt(mse))
    ssim_value = ssim(
        img1, img2, channel_axis=2, data_range=img2.max() - img2.min()
    )  # Ensure data_range is correctly set

    return mse, psnr_value, ssim_value


def plot_metrics(metrics_figure4, metrics_figure5, save_path="metrics_plot.png"):
    """
    Plots MSE, PSNR, and SSIM metrics for figure4 and figure5.

    Args:
        metrics_figure4 (tuple): MSE, PSNR, and SSIM values for figure4.
        metrics_figure5 (tuple): MSE, PSNR, and SSIM values for figure5.
        save_path (str): Path to save the plot.
    """
    metrics = ["MSE", "PSNR", "SSIM"]
    figure4_values = metrics_figure4
    figure5_values = metrics_figure5

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, figure4_values, width, label="Figure 4")
    rects2 = ax.bar(x + width / 2, figure5_values, width, label="Figure 5")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Values")
    ax.set_title("Metrics for Figure 4 and Figure 5")
    ax.set_xticks(x, metrics)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Set the base directories
    figure4_dir = "flow_interpolate_plot_ae/Ioannis/figure4"
    figure5_dir = "flow_interpolate_plot_ae/Ioannis/figure5"

    # Set the naming patterns
    figure4_gt_pattern = "{}_{}.png"
    figure4_rec_pattern = "{}_rec_{}.png"
    figure5_gt_pattern = "{}_{}.png"
    figure5_rec_pattern = "{}_rec_midpoint_{}.png"

    # List of numbers to iterate through
    numbers = ["1104", "1128", "1152", "1176", "1200", "1224"]
    qualities = ["coarse", "medium", "fine"]

    # Initialize dictionaries to store metrics for each quality
    figure4_metrics = {
        quality: {"mse": [], "psnr": [], "ssim": []} for quality in qualities
    }
    figure5_metrics = {
        quality: {"mse": [], "psnr": [], "ssim": []} for quality in qualities
    }

    # Iterate through the images and calculate metrics
    for number in numbers:
        for quality in qualities:
            # Create the file paths
            figure4_gt_path = os.path.join(
                figure4_dir, figure4_gt_pattern.format(number, quality)
            )
            figure4_rec_path = os.path.join(
                figure4_dir, figure4_rec_pattern.format(number, quality)
            )
            figure5_gt_path = os.path.join(
                figure5_dir, figure5_gt_pattern.format(number, quality)
            )
            figure5_rec_path = os.path.join(
                figure5_dir, figure5_rec_pattern.format(number, quality)
            )

            # Calculate metrics
            try:
                figure4_mse, figure4_psnr, figure4_ssim = calculate_metrics(
                    figure4_gt_path, figure4_rec_path
                )

                # Store metrics
                figure4_metrics[quality]["mse"].append(figure4_mse)
                figure4_metrics[quality]["psnr"].append(figure4_psnr)
                figure4_metrics[quality]["ssim"].append(figure4_ssim)

                print(
                    f"Figure4 ({number}, {quality}): MSE={figure4_mse:.2f}, PSNR={figure4_psnr:.2f}, SSIM={figure4_ssim:.2f}"
                )

                if "medium" in figure5_rec_path:
                    figure5_mse, figure5_psnr, figure5_ssim = calculate_metrics(
                        figure5_gt_path, figure5_rec_path
                    )
                    figure5_metrics[quality]["mse"].append(figure5_mse)
                    figure5_metrics[quality]["psnr"].append(figure5_psnr)
                    figure5_metrics[quality]["ssim"].append(figure5_ssim)
                    print(
                        f"Figure5 ({number}, {quality}): MSE={figure5_mse:.2f}, PSNR={figure5_psnr:.2f}, SSIM={figure5_ssim:.2f}"
                    )

            except FileNotFoundError:
                print(f"Skipping {number} due to missing files.")
                continue

    # Calculate the average metrics for each quality
    avg_figure4_metrics = {}
    avg_figure5_metrics = {}
    for quality in qualities:
        avg_figure4_metrics[quality] = {
            "mse": np.mean(figure4_metrics[quality]["mse"]),
            "psnr": np.mean(figure4_metrics[quality]["psnr"]),
            "ssim": np.mean(figure4_metrics[quality]["ssim"]),
        }
        avg_figure5_metrics[quality] = {
            "mse": np.mean(figure5_metrics[quality]["mse"]),
            "psnr": np.mean(figure5_metrics[quality]["psnr"]),
            "ssim": np.mean(figure5_metrics[quality]["ssim"]),
        }

        print(
            f"Figure4 ({quality}): MSE={avg_figure4_metrics[quality]['mse']:.2f}, PSNR={avg_figure4_metrics[quality]['psnr']:.2f}, SSIM={avg_figure4_metrics[quality]['ssim']:.2f}"
        )
        print(
            f"Figure5 ({quality}): MSE={avg_figure5_metrics[quality]['mse']:.2f}, PSNR={avg_figure5_metrics[quality]['psnr']:.2f}, SSIM={avg_figure5_metrics[quality]['ssim']:.2f}"
        )

    # Plot the average metrics
    # plot_metrics(
    #     (avg_figure4_mse, avg_figure4_psnr, avg_figure4_ssim),
    #     (avg_figure5_mse, avg_figure5_psnr, avg_figure5_ssim),
    # )
