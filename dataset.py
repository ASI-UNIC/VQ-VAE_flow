from torch.utils.data import Dataset, TensorDataset
import torch
from PIL import Image
import numpy as np
import random
import h5py
import os
from torchvision import transforms


class AllImageDataset(Dataset):
    def __init__(self, h5py_file: str, test_size=0.1):
        np.random.seed(42)
        random.seed(42)
        self.flow_images = h5py.File(h5py_file, "r")[
            "flow"
        ]  # [batch, height, width, channel]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def denormalize(self, x):
        if x.shape[-1] != 3:
            reshaped_x = x.permute(0, 2, 3, 1)
        return reshaped_x

    def __len__(self):
        return len(self.flow_images)

    def __getitem__(self, index):
        return self.transforms(self.flow_images[index])


class InterpolateDataset(Dataset):
    def __init__(self, h5py_file: str, test_size=0.1):
        np.random.seed(42)
        random.seed(42)
        self.flow_images = h5py.File(h5py_file, "r")[
            "flow"
        ]  # [batch, height, width, channel]
        self.intermediate_flow_images = h5py.File(h5py_file, "r")[
            "intermediate_flow"
        ]  # [batch, height, width, channel]
        self.target_flow_images = h5py.File(h5py_file, "r")[
            "target_flow"
        ]  # [batch, height, width, channel]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def denormalize(self, x):
        if x.shape[-1] != 3:
            reshaped_x = x.permute(0, 2, 3, 1)
        return reshaped_x

    def __len__(self):
        return len(self.flow_images)

    def __getitem__(self, index):
        return (
            self.transforms(self.flow_images[index]),
            self.transforms(self.intermediate_flow_images[index]),
            self.transforms(self.target_flow_images[index]),
        )


class InterpolateIntermediateDataset(InterpolateDataset):
    def __init__(self, h5py_file, test_size=0.1, is_train=True):
        super().__init__(h5py_file, test_size)
        train_length = int(len(self.flow_images) * (1 - test_size))

        if is_train:
            self.flow_images = self.flow_images[:train_length]
            self.intermediate_flow_images = self.intermediate_flow_images[:train_length]
            self.target_flow_images = self.target_flow_images[:train_length]
        else:
            self.flow_images = self.flow_images[train_length:]
            self.intermediate_flow_images = self.intermediate_flow_images[train_length:]
            self.target_flow_images = self.target_flow_images[train_length:]


class AllTargetImageDataset(Dataset):
    def __init__(self, h5py_file: str, test_size=0.1):
        np.random.seed(42)
        random.seed(42)
        self.flow_images = h5py.File(h5py_file, "r")[
            "flow"
        ]  # [batch, height, width, channel]
        self.target_flow_images = h5py.File(h5py_file, "r")[
            "target_flow"
        ]  # [batch, height, width, channel]
        # self.mean = [0.5, 0.5, 0.5]
        # self.std = [0.5, 0.5, 0.5]
        self.transforms = transforms.Compose([transforms.ToTensor()])  # ,
        # transforms.Normalize(self.mean ,self.std)])

    def denormalize(self, x):
        if x.shape[-1] != 3:
            reshaped_x = x.permute(0, 2, 3, 1)
        return reshaped_x
        # return  reshaped_x * np.array(self.std) + np.array(self.mean)

    def __len__(self):
        return len(self.target_flow_images)

    def __getitem__(self, index):
        return self.transforms(self.flow_images[index]), self.transforms(
            self.target_flow_images[index]
        )


class LatentDataset(AllImageDataset):
    def __init__(self, h5py_file: str, test_size=0.1):
        np.random.seed(42)
        random.seed(42)
        h5_content = h5py.File(h5py_file, "r")
        self.flow_latents = h5_content["flow"]
        self.target_flow_latents = h5_content["target_flow"]

    def __len__(self):
        return len(self.flow_latents)

    def __getitem__(self, index):
        return self.flow_latents[index], self.target_flow_latents[index]


class ImageDataset(Dataset):
    def __init__(self, image_folder, target_image_folder, test_size=0.1):
        np.random.seed(42)
        random.seed(42)

        self.images, filenames = self.load_images_from_folder(
            image_folder
        )  # [batch, height, width, channel]
        self.target_images, target_filenames = self.load_images_from_folder(
            target_image_folder
        )  # [batch, height, width, channel]

        # get intersecting filenames only
        indices_filenames, indices_target_filenames = self.__get_intersect_names(
            filenames, target_filenames
        )
        self.images = self.images[indices_filenames]
        self.target_images = self.target_images[indices_target_filenames]

        # shuffle the images and target images
        indices = np.arange(self.images.shape[0])
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.target_images = self.target_images[indices]

        # get training and testing set
        self.train_images = self.images[: int(len(self.images) * (1 - test_size))]
        self.test_images = self.images[int(len(self.images) * (1 - test_size)) :]
        self.train_target_images = self.target_images[
            : int(len(self.target_images) * (1 - test_size))
        ]
        self.test_target_images = self.target_images[
            int(len(self.target_images) * (1 - test_size)) :
        ]

        self.train_dataset = TensorDataset(
            torch.FloatTensor(self.train_images),
            torch.FloatTensor(self.train_target_images),
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(self.test_images),
            torch.FloatTensor(self.test_target_images),
        )

    def __get_intersect_names(self, filenames, target_filenames):
        filenames = [filename.split(".")[1] for filename in filenames]
        target_filenames = [filename.split(".")[1] for filename in target_filenames]

        common_strings = set(filenames) & set(target_filenames)
        sorted(common_strings)
        # Get indices of common strings in both lists
        indices_filenames = [i for i, s in enumerate(filenames) if s in common_strings]
        indices_target_filenames = [
            i for i, s in enumerate(target_filenames) if s in common_strings
        ]

        # filenames are in the same order
        return indices_filenames, indices_target_filenames

    def load_images_from_folder(self, image_folder):
        images = []
        filenames = []  # need this so we can match with the target image
        for filename in os.listdir(image_folder):
            filenames.append(filename)

            img_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")  # Load and convert to RGB
                img = img.resize((512, 256))  # Resize image ori is 1024, 512
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        images = np.array(images)
        images = images.transpose(0, 3, 1, 2).astype(
            np.float32
        )  # [batch, channel, height, width]

        return images, filenames

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index]
