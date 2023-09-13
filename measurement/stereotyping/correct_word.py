import os

import numpy as np
import pandas as pd
import tqdm
from sklearn.linear_model import LogisticRegression

from measurement.loading import load_coco_annotations, load_visual_genome_labels, \
    load_scene_graphs_in_model_captions, load_zhao_image_paths


def load_annotations():
    # load human-labeled image annotations, (These should include locations/size of all images)
    coco_annotations = load_coco_annotations()

    coco_image_observations = load_coco_annotations(include_size_and_location=True)
    vg_image_observations, _, _ = list(load_visual_genome_labels(include_size_and_location=True))

    # Unload synset lists from visual genome
    i = 0
    all_na = False
    while not all_na:
        vg_image_observations[f'syn_{i}'] = vg_image_observations['synset'].str[i]
        all_na = vg_image_observations[f'syn_{i}'].isna().all()
        i += 1
    vg_image_observations = vg_image_observations.melt(id_vars=['image_id', 'size', 'distance'],
                                                       value_vars=[f'syn_{k}' for k in range(i)])
    vg_image_observations = vg_image_observations.drop(columns='variable')
    vg_image_observations = vg_image_observations[~vg_image_observations['value'].isna()]
    vg_image_observations = vg_image_observations.rename(columns={'value': 'synset'})

    # Join vg/coco annotations
    image_observations = pd.concat([coco_image_observations, vg_image_observations]).drop_duplicates()

    # Filter down to images used by Zhao et al.
    zhao_image_paths = load_zhao_image_paths()
    image_observations = image_observations[image_observations['image_id'].isin(zhao_image_paths['image_id'])]
    return image_observations


def calculate_most_common_objects(model_name, image_observations):
    """
    Load the 500 most common object types, across visual genome, the captions output by the model, the coco annotations.

    :param model_name:
    :param image_observations:
    :return:
    """

    # Calculate the 500 most common object types in stages 1 and 4
    caption_objs, _, _ = load_scene_graphs_in_model_captions(model_name, 'coco')
    stage_1_and_4_together = pd.concat([image_observations, caption_objs])

    most_common_objects = stage_1_and_4_together['synset'].value_counts().head(500).index.tolist()
    return most_common_objects


def label_images():
    """
    For each object type, find all images that contain that object type according to stage 1, and label them a
    false negative if they are not labeled as containing that object type in stage 4, or a true positive if they are

    :return:
    """
    caption_objs, _, _ = load_scene_graphs_in_model_captions(model_name, 'coco')

    ground_truth = image_observations[['image_id', 'synset']].drop_duplicates()
    predictions = caption_objs[['image_id', 'synset']].drop_duplicates()
    predictions['is_contained_in_predictions'] = True
    labels = pd.merge(ground_truth, predictions, on=['image_id', 'synset'], how='left').fillna(False)

    return labels


def test_train_splits():
    """
    Considering all images that contain the object of interest, create 1000 test-train splits (70-30)

    :return:
    """

    np.random.seed(926056)
    test_train_splits = []
    with tqdm.tqdm(total=len(most_common_objects) * 1000) as pbar:
        for obj in most_common_objects:
            object_images = labels[labels['synset'] == obj]
            split_loc = int(len(object_images) * .7)
            for i in range(1000):
                object_images = object_images.sample(frac=1, replace=False)
                contains_tp_and_fn = len(
                    object_images.iloc[:split_loc]['is_contained_in_predictions'].value_counts()) == 2
                train_ids = object_images['image_id'].iloc[:split_loc].tolist()
                test_ids = object_images['image_id'].iloc[split_loc:].tolist()
                test_train_splits.append(pd.DataFrame({'object': [obj], 'train_ids': [train_ids],
                                                       'test_ids': [test_ids],
                                                       'contains_tp_and_fn': [contains_tp_and_fn]}))
                pbar.update(1)

    test_train_splits = pd.concat(test_train_splits)

    # Throw out any test-train splits that do not have both true positives and false negatives
    test_train_splits = test_train_splits[test_train_splits['contains_tp_and_fn']]
    test_train_splits = test_train_splits.drop(columns='contains_tp_and_fn')

    # (If there are less than 900 splits remaining, throw out the object type)
    split_counts = test_train_splits['object'].value_counts()
    objs_to_keep = split_counts[split_counts > 900].index.tolist()
    test_train_splits = test_train_splits[test_train_splits['object'].isin(objs_to_keep)]

    return test_train_splits


def load_data_for_regressions(image_observations):
    """
    # Get size and location of all objects in each image; Since there can be multiple objects of the same type in an image,
    # take the largest object of each type in each image #TODO: Figure out if there's a better way to handle multiple objects

    :param image_observations:
    :return:
    """
    image_observations = image_observations.sort_values('size', ascending=False)
    observed_objects = image_observations.groupby(['image_id', 'synset']).head(1)
    observed_objects = observed_objects[observed_objects.synset.isin(most_common_objects)]
    sizes = observed_objects.pivot(index='image_id', columns='synset', values=['size']).fillna(
        0)  # TODO: How should we handle missing values?
    distances = observed_objects.pivot(index='image_id', columns='synset', values=['distance']).fillna(
        1)  # TODO: How should we handle missing values?
    sizes.columns = sizes.columns.droplevel(0) + '_size'
    distances.columns = distances.columns.droplevel(0) + '_distance'
    image_features = pd.concat([sizes, distances],
                               axis=1)  # Todo: There are less than 1,000 columns, which is different from what is stated in paper, so need to double check

    # Get race/gender features for each image
    demographic_data = pd.read_csv(os.path.join('input_data', 'COCO 2014 Val Demographic Annotations',
                                                'images_val2014.csv'))
    demographic_data = demographic_data.rename(columns={'id': 'image_id'})[['image_id', 'bb_skin', 'bb_gender']]
    demographic_data['bb_skin'] = demographic_data['bb_skin'] == 'Light'
    demographic_data['bb_gender'] = demographic_data['bb_gender'] == 'Male'
    image_features = pd.merge(image_features, demographic_data, on='image_id', how='left')

    return image_features


def train_classifiers(test_train_splits, image_features, labels):
    # Train a classifier to predict whether an image contains the object of interest based on the size and location of all
    # 500 objects in the image, as well as the social characteristic of the person contained
    skin_regressions = []
    skin_coefs = []
    skin_accuracies = []
    gender_regressions = []
    gender_coefs = []
    gender_accuracies = []
    with tqdm.tqdm(total=len(test_train_splits)) as pbar:
        for i, (object, train_ids, test_ids) in test_train_splits.iterrows():
            train_set = image_features[image_features['image_id'].isin(train_ids)].sort_values('image_id')
            train_labels = labels[labels['synset'] == object].merge(train_set[['image_id']], on='image_id',
                                                                    how='inner').sort_values('image_id')
            test_set = image_features[image_features['image_id'].isin(test_ids)].sort_values('image_id')
            test_labels = labels[labels['synset'] == object].merge(test_set[['image_id']], on='image_id',
                                                                   how='inner').sort_values('image_id')

            # Filter out any degenerate columns
            degenerate_columns = [c for c in train_set.columns if len(train_set[c].value_counts()) == 1]
            train_set = train_set.drop(columns=degenerate_columns)
            test_set = test_set.drop(columns=degenerate_columns)

            skin_regressions.append(LogisticRegression(max_iter=1000).fit(
                train_set.drop(columns=['image_id', 'bb_gender']).values,
                train_labels['is_contained_in_predictions'].values,
            ))
            skin_coefs.append(skin_regressions[-1].coef_[0, -1])
            skin_accuracies.append(skin_regressions[-1].score(
                test_set.drop(columns=['image_id', 'bb_gender']).values,
                test_labels['is_contained_in_predictions'].values,
            ))
            gender_regressions.append(LogisticRegression(max_iter=1000).fit(
                train_set.drop(columns=['image_id', 'bb_skin']).values,
                train_labels['is_contained_in_predictions'].values,
            ))
            gender_coefs.append(gender_regressions[-1].coef_[0, -1])
            gender_accuracies.append(gender_regressions[-1].score(
                test_set.drop(columns=['image_id', 'bb_skin']).values,
                test_labels['is_contained_in_predictions'].values,
            ))
            pbar.update(1)

    test_train_splits['skin_regression'] = skin_regressions
    test_train_splits['skin_coef'] = skin_coefs
    test_train_splits['skin_accuracy'] = skin_accuracies
    test_train_splits['gender_regression'] = gender_regressions
    test_train_splits['gender_coef'] = gender_coefs
    test_train_splits['gender_accuracy'] = gender_accuracies
    pd.to_csv(f'processed_data/measurements/correct_word/{model_name}.csv', index=False)


if __name__ == '__main__':
    image_observations = load_annotations()
    image_features = load_data_for_regressions(image_observations)

    for model_name in ['clipcap_coco_mlp', 'clipcap_coco_transformer','clipcap_conceptual_mlp']:
        most_common_objects = calculate_most_common_objects('clipcap_coco_mlp', image_observations)
        labels = label_images()
        test_train = test_train_splits()
        train_classifiers(test_train, image_features, labels)
