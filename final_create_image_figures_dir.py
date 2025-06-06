import os
import shutil


def main(flow_directory: str, output_directory: str):
    os.makedirs(output_directory, exist_ok=True)

    images = ["1104", "1128", "1152", "1176", "1200", "1224"]
    images_selection = [
        "0_result.png",
        "2_result.png",
        "4_result.png",
        "6_result.png",
        "8_result.png",
        "10_result.png",
        "rec_data.png",
        "rec_intermediate.png",
        "rec_target.png",
    ]

    new_images_name = [
        "0.png",
        "2.png",
        "4.png",
        "6.png",
        "8.png",
        "10.png",
        "coarse.png",
        "medium.png",
        "fine.png",
    ]

    for idx, image in enumerate(images):
        current_dir = os.path.join(flow_directory, f"interpolation_{idx}")

        for selected_image, new_image_name in zip(images_selection, new_images_name):
            filename = os.path.join(current_dir, selected_image)

            shutil.copy(
                filename,
                os.path.join(output_directory, f"{image}_{new_image_name}"),
            )


if __name__ == "__main__":
    main("flow_interpolate_plot_uz", "flow_interpolate_plot_uz_figure8")
