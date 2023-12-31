import json
import os

import pandas as pd

from measurement.loading import load_zhao_image_paths


def load_images_remaining(model_name):
    """
    Load the list of images that have not yet been captioned by a model.

    :param model_name: Name of model that will be used to generate captions
    :return:
    """
    complete_list_of_image_paths = load_zhao_image_paths()
    save_path = os.path.join('processed_data','coco_captions', f'{model_name}.csv')
    if not os.path.exists(save_path):
        return complete_list_of_image_paths
    else:
        completed_image_paths = pd.read_csv(save_path)
        outer_join = completed_image_paths.merge(complete_list_of_image_paths, on='image_id', how='outer', indicator=True)
        image_paths = outer_join[~(outer_join._merge == 'both')].drop(['_merge','caption'], axis=1)
        return image_paths


def save_image_caption(model_name, captions_to_save):
    """
    Save the captions generated by a model to a csv file. Will append them to the existing file if it exists.

    :param model_name: Name of model used to generate captions
    :param captions_to_save: Dataframe with columns 'image_id' and 'caption'
    :return:
    """
    save_path = os.path.join('processed_data', 'coco_captions', f'{model_name}.csv')
    if not os.path.exists(save_path):
        captions_to_save.to_csv(save_path, index=False)
    else:
        completed_image_paths = pd.read_csv(save_path)
        completed_image_paths = pd.concat([completed_image_paths, captions_to_save])
        completed_image_paths.to_csv(save_path, index=False)

def load_human_generated_captions():
    """
    Load the human-generated captions of the COCO dataset.

    :return:
    """
    captions = json.load(open(os.path.join('input_data','annotations','captions_val2014.json')))
    return captions


