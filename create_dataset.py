import torch
import h5py
from glob import glob
import os
from PIL import Image
import numpy as np
import click
from tqdm import tqdm
from unet_lightning import RVQUNetLightning

batch_size = 32

width = 64
height = 32
channel = 3

def get_intersect_names(*array_filenames):
        
        processed_filenames = []
        for filenames in array_filenames:
            filenames = [filename.split('.')[1] for filename in filenames]
            processed_filenames.append(filenames)

        common_strings = set(processed_filenames[0])
        for processed_filename in processed_filenames[1:]:
            common_strings = common_strings & set(processed_filename)
        common_strings = list(common_strings)
        common_strings = sorted(common_strings)

        # Get indices of common strings in both lists
        indices_filenames = []
        for filenames in processed_filenames:
            indices = [filenames.index(s) for s in common_strings if s in filenames]
            indices_filenames.append(indices)
        # filenames are in the same order
        return indices_filenames


def preprocess_image(img: np.array):
    img = np.array(img.resize((width, height)))  # Resize image ori is 1024, 512
    return img


def create_h5_dataset(filenames: str, output_filepath: str, width: int, height: int,channel: int, batch_size: int, h5_name: str = 'flow'):
    with h5py.File(output_filepath, 'a') as hf:

        for i in tqdm(range(0, len(filenames), batch_size)):
            current_filenames = filenames[i:i+batch_size]

            all_img_array = []
            for filename in current_filenames:
                img = Image.open(filename).convert('RGB')  # Load and convert to RGB
                img_array = preprocess_image(img)
                all_img_array.append(img_array)
            flow = np.array(all_img_array)

            if i == 0:
                hf.create_dataset(h5_name, data=flow, chunks=True, maxshape=(None, height, width, channel))
            else:
                hf[h5_name].resize((hf[h5_name].shape[0] + flow.shape[0]), axis = 0)
                hf[h5_name][-flow.shape[0]:] = flow


def create_h5_latent_dataset(filenames: str, output_filepath: str, model_path: str, width: int, height: int,channel: int, batch_size: int, h5_name: str = 'latent'):

    # load the model
    model_filename = os.path.join(model_path, 'works-rvq-unet.ckpt')
    model_config = os.path.join(model_path, 'config.json')
    config = torch.load(model_config)
    n_channels = config['n_channels']
    n_embeddings = config['num_embeddings']
    num_quantizers = config['num_quantizers']
    max_epochs = config['total_epochs']
    resvq_embeddings = config['resvq_embeddings']
    codebook_size = config['codebook_size']
    model = RVQUNetLightning.load_from_checkpoint(model_filename, n_channels=n_channels, num_embeddings=n_embeddings, total_epochs=max_epochs,
                                num_quantizers=num_quantizers, resvq_embeddings=resvq_embeddings, codebook_size=codebook_size)

    with h5py.File(output_filepath, 'a') as hf:

        for i in tqdm(range(0, len(filenames), batch_size)):
            current_filenames = filenames[i:i+batch_size]

            all_img_arrays = []
            for filename in current_filenames:
                img = Image.open(filename).convert('RGB')  # Load and convert to RGB
                img_array = preprocess_image(img)
                img_array = img_array.transpose(2,0,1)
                all_img_arrays.append(img_array)
            
            all_img_arrays = torch.Tensor(np.array(all_img_arrays))
            all_latents = model.get_latent(all_img_arrays).detach().cpu().numpy()

            if i == 0:
                hf.create_dataset(h5_name, data=all_latents, chunks=True, maxshape=(None, *all_latents.shape[1:]))
            else:
                hf[h5_name].resize((hf[h5_name].shape[0] + all_latents.shape[0]), axis = 0)
                hf[h5_name][-all_latents.shape[0]:] = all_latents

def create_pretrain_dataset(image_folder: str, target_folder: str, output_filepath: str, train_type='not_all'):
    filenames = glob(os.path.join(image_folder, "**/*.png"), recursive=True)
    target_filenames = glob(os.path.join(target_folder, "**/*.png"), recursive=True)
    # get intersecting filenames only

    if train_type != 'all':
        indices_filenames, indices_target_filenames = get_intersect_names(filenames, target_filenames)
        filenames = np.array(filenames)[indices_filenames]
        target_filenames = np.array(target_filenames)[indices_target_filenames]

        create_h5_dataset(filenames, output_filepath, width, height, channel, batch_size, 'flow')
        create_h5_dataset(target_filenames, output_filepath, width, height, channel, batch_size, 'target_flow')
    else:
        all_filenames = filenames + target_filenames
        create_h5_dataset(all_filenames, output_filepath, width, height, channel, batch_size, 'flow')


def create_interpolate_dataset(image_folder, intermediate_folder, target_folder, output_filepath):
    filenames = glob(os.path.join(image_folder, "**/*.png"), recursive=True)
    intermediate_filenames = glob(os.path.join(intermediate_folder, "**/*.png"), recursive=True)
    target_filenames = glob(os.path.join(target_folder, "**/*.png"), recursive=True)

    indices_filenames,indices_intermediate_filenames, indices_target_filenames = get_intersect_names(filenames, intermediate_filenames, target_filenames)
    filenames = np.array(filenames)[indices_filenames]
    intermediate_filenames = np.array(intermediate_filenames)[indices_intermediate_filenames]
    target_filenames = np.array(target_filenames)[indices_target_filenames]

    create_h5_dataset(filenames, output_filepath, width, height, channel, batch_size, 'flow')
    create_h5_dataset(intermediate_filenames, output_filepath, width, height, channel, batch_size, 'intermediate_flow')
    create_h5_dataset(target_filenames, output_filepath, width, height, channel, batch_size, 'target_flow')

def create_train_dataset(image_folder: str, target_folder: str, model_path: str, output_filepath: str, test_size:float = 0.1):
    filenames = glob(os.path.join(image_folder, "**/*.png"), recursive=True)
    target_filenames = glob(os.path.join(target_folder, "**/*.png"), recursive=True)

    # get intersecting filenames only
    indices_filenames, indices_target_filenames = get_intersect_names(filenames, target_filenames)
    filenames = np.array(filenames)[indices_filenames]
    target_filenames = np.array(target_filenames)[indices_target_filenames]

    create_h5_latent_dataset(filenames, output_filepath, model_path, width, height, channel, batch_size, 'flow')
    create_h5_latent_dataset(target_filenames, output_filepath, model_path, width, height, channel, batch_size, 'target_flow')

    # create train and test dataset
    flow_latents = h5py.File(output_filepath, 'r')['flow']
    target_flow_latents = h5py.File(output_filepath, 'r')['target_flow']

    train_flow_latents = flow_latents[:int(len(flow_latents) * (1 - test_size))]
    test_flow_latents = flow_latents[int(len(flow_latents) * (1 - test_size)):]

    train_target_flow_latents = target_flow_latents[:int(len(target_flow_latents) * (1 - test_size))]
    test_target_flow_latents = target_flow_latents[int(len(target_flow_latents) * (1 - test_size)):]

    train_filename = os.path.join(os.path.dirname(output_filepath), 'train_' + os.path.basename(output_filepath))
    test_filename = os.path.join(os.path.dirname(output_filepath), 'test_' + os.path.basename(output_filepath))
    with h5py.File(train_filename, 'a') as hf:
        hf.create_dataset('flow', data=train_flow_latents, chunks=True, maxshape=(None, *train_flow_latents.shape[1:]))
        hf.create_dataset('target_flow', data=train_target_flow_latents, chunks=True, maxshape=(None, *train_target_flow_latents.shape[1:]))
    with h5py.File(test_filename, 'a') as hf:
        hf.create_dataset('flow', data=test_flow_latents, chunks=True, maxshape=(None, *test_flow_latents.shape[1:]))
        hf.create_dataset('target_flow', data=test_target_flow_latents, chunks=True, maxshape=(None, *test_target_flow_latents.shape[1:]))
    

    

@click.command()
@click.option(
    '--image_folder',
    required=True,
    type=str,)
@click.option(
    '--intermediate_folder',
    type=str,)
@click.option(
    '--target_folder',
    type=str
)
@click.option('--model_path', type=str, help="Path to the trained model for mapping between flow to target flow latents")
@click.option(
    '--output_filepath',
    required=True,
    type=str
)
@click.option(
    '--dataset_type',
    required=True,
    default='pretrain',
    type=click.Choice(['pretrain', 'interpolate', 'train'])
)
def create_dataset(dataset_type: str, image_folder: str, model_path:str, intermediate_folder: str, target_folder: str, output_filepath: str):

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    dataset_dict = {
        'pretrain': lambda image_folder=image_folder, output_filepath=output_filepath: create_pretrain_dataset(image_folder, target_folder, output_filepath),
        'interpolate': lambda image_folder=image_folder, intermediate_folder=intermediate_folder, target_folder=target_folder, output_filepath=output_filepath: create_interpolate_dataset(image_folder, intermediate_folder, target_folder, output_filepath),
        'train': lambda image_folder=image_folder, target_folder=target_folder, model_path=model_path, output_filepath=output_filepath: create_train_dataset(image_folder, target_folder, model_path, output_filepath)
    }

    dataset_dict[dataset_type]()


if __name__ == '__main__':
    create_dataset()