import json
import os
import re
import subprocess
from ast import literal_eval
from io import StringIO

import nltk
import numpy as np
import pandas as pd

from measurement.utils import most_common_synset


def parse_single_scene(scene_text, index):
    """
    Parse a single scene from the output of the Stanford scene graph parser

    :param scene_text: output of the Stanford scene graph parser
    :param index: image_id to associate with this scene
    :return:
    """
    relations_table, nodes_table, _ = scene_text.split('\n\n')

    relations_table = pd.read_csv(StringIO(relations_table), sep='\s{2,}', engine='python')
    if '> source' in relations_table.columns:
        relations_table = relations_table.rename(columns={'> source': 'source'})
    relations_table = relations_table[~relations_table['source'].str.contains('--')].reset_index(drop=True)
    relations_table['image_id'] = index

    # Since a node can have multiple attributes, we need to parse them separately
    attributes = re.findall(r'\n([^ ]+) +\n +-([^ ]+) *', nodes_table)
    new_attributes = attributes
    i = 0
    while len(new_attributes) > 0:
        i += 1
        unused = '(?:\n +-[^ ]+ *)' * i
        new_attributes = re.findall(rf'\n([^ ]+) +{unused}\n +-([^ ]+) *', nodes_table)
        attributes += new_attributes
    attributes_table = pd.DataFrame(attributes, columns=['object_id', 'attribute'])
    attributes_table['image_id'] = index

    nodes_table = pd.read_csv(StringIO(nodes_table), sep='\s{2,}', engine='python')
    nodes_table = nodes_table[~nodes_table['Nodes'].str.contains('--')].reset_index(drop=True)
    nodes_table = nodes_table.rename(columns={'Nodes': 'object_id'})
    nodes_table = nodes_table[nodes_table['object_id'].str.slice(0, 1) != '-'].reset_index(drop=True)
    nodes_table['image_id'] = index

    # Sometimes the relation and target get squished together, so this just splits them when necessary
    if relations_table['reln'].str.contains('\d', regex=True).any():
        relations_table['target'] = np.where(
            relations_table['target'].isna(),
            relations_table['reln'].str.replace('.+(' + '|'.join(nodes_table['object_id']) + ')', '\\1', regex=True),
            relations_table['target']
        )
        relations_table['reln'] = relations_table['reln'].str.replace('|'.join(nodes_table['object_id']), '',
                                                                      regex=True).str.strip()

    return relations_table, nodes_table, attributes_table


def load_scene_graphs_in_model_captions(model_name, dataset_name):
    """
    Load scene graphs for each caption output by a model. If the scene graphs have already been created, load them from
    disk. Otherwise, run the Stanford scene graph parser on the captions.

    :param model_name: Name of model that was used to make the captions
    :param dataset_name: Name of dataset that was used to make the captions
    :return:
    """
    relations_path = os.path.join(f'processed_data', f'{dataset_name}_scene_graphs', f'{model_name}_relations.csv')
    nodes_path = os.path.join(f'processed_data', f'{dataset_name}_scene_graphs', f'{model_name}_nodes.csv')
    attributes_path = os.path.join(f'processed_data', f'{dataset_name}_scene_graphs', f'{model_name}_attributes.csv')

    if os.path.exists(relations_path) and os.path.exists(nodes_path) and os.path.exists(attributes_path):
        relations = pd.read_csv(relations_path)
        nodes = pd.read_csv(nodes_path)
        attributes = pd.read_csv(attributes_path)
        return nodes, relations, attributes
    else:
        captions = pd.read_csv(os.path.join(f'processed_data', f'{dataset_name}_captions', f'{model_name}.csv'))

        # Run Stanford scene graph parser
        to_parse = '\n'.join(captions['caption'])
        str_input = to_parse.encode('utf-8')
        classpath = os.path.join(os.getcwd(), 'references', 'stanford-corenlp-full-2015-12-09', '*')
        result = subprocess.Popen(
            ['java', '-mx2g', '-cp', f'.:{classpath}', 'edu.stanford.nlp.scenegraph.RuleBasedParser'],
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = result.communicate(input=str_input)
        out = out[0].decode('utf-8')
        # Remove header and footer
        scenes = out.split('INFO: Read 25 rules')[1]

        # Split into separate scenes
        scenes = scenes.split('------------------------')[:-1]

        # Parse each scene
        scenes = [parse_single_scene(s, i) for s, i in zip(scenes, captions['image_id'])]

        relations = pd.concat([s[0] for s in scenes])
        nodes = pd.concat([s[1] for s in scenes])
        attributes = pd.concat([s[2] for s in scenes])

        # Get synsets in objects
        nodes['node_name'] = nodes['object_id'].str.split('-').str[0]
        mapper = {w: most_common_synset(w, 'n').name() for w in np.unique(nodes['node_name'])
                  if most_common_synset(w, 'n') is not None}
        for w in np.unique(nodes['node_name']):
            if w not in mapper.keys():
                mapper[w] = None
        nodes['synset'] = nodes['node_name'].map(mapper)

        # Get synsets in attributes
        mapper = {w: most_common_synset(w, 'a').name() for w in np.unique(attributes['attribute'])
                  if most_common_synset(w, 'a') is not None}
        for w in np.unique(attributes['attribute']):
            if w not in mapper.keys():
                mapper[w] = None
        attributes['synset'] = attributes['attribute'].map(mapper)

        # TODO: HANDLE MULTI-WORD RELATIONS. RIGHT NOW, EACH ONE GETS ITS OWN ROW
        # Get synsets in relationships
        words_in_relns = {reln: nltk.word_tokenize(reln) for reln in np.unique(relations['reln'])}
        for reln, words in words_in_relns.items():
            words_in_relns[reln] = [most_common_synset(w, ['v', 'r']).name() for w in words if
                                    most_common_synset(w, ['v', 'r']) is not None]
        relations_full = []
        for i, row in relations.iterrows():
            for syn in words_in_relns[row['reln']]:
                relations_full.append([row['source'], row['reln'], syn, row['target'], row['image_id']])
            if len(words_in_relns[row['reln']]) == 0:
                relations_full.append([row['source'], row['reln'], None, row['target'], row['image_id']])
        relations = pd.DataFrame(relations_full, columns=['source', 'reln', 'synset', 'target', 'image_id'])
        relations = relations.drop_duplicates().reset_index(drop=True)

        relations.to_csv(relations_path, index=False)
        nodes.to_csv(nodes_path, index=False)
        attributes.to_csv(attributes_path, index=False)

        return nodes, relations, attributes


def load_coco_annotations(include_size_and_location=False):
    """
    Load human annotations describing which objects are in each image of the COCO dataset.

    If requested, size and location information is calculated, as described in "Understanding and predicting importance
    in images" by Berg et al.

    :param include_size_and_location: Whether to include size and location information for the objects.
    :return:
    """
    coco_annotations = json.load(open(os.path.join('input_data', 'annotations', 'instances_val2014.json')))
    categories = coco_annotations['categories']
    width_map = {i['id']: i['width'] for i in coco_annotations['images']}
    height_map = {i['id']: i['height'] for i in coco_annotations['images']}
    coco_annotations = coco_annotations['annotations']

    if not include_size_and_location:
        coco_annotations = [pd.DataFrame({'image_id': [o['image_id']], 'category_id': o['category_id']})
                            for o in coco_annotations]
        image_observations = pd.concat(coco_annotations)

    else:
        coco_annotations = [pd.DataFrame({'image_id': [o['image_id']], 'category_id': o['category_id'],
                                          'xmin': [o['bbox'][0]], 'ymin': [o['bbox'][1]],
                                          'width': [o['bbox'][2]], 'height': [o['bbox'][3]],
                                          'area': [o['area']]})
                            for o in coco_annotations]
        image_observations = pd.concat(coco_annotations)

        image_observations['image_width'] = image_observations['image_id'].map(width_map)
        image_observations['image_height'] = image_observations['image_id'].map(height_map)
        image_observations['size'] = (image_observations['area'] /
                                      (image_observations['image_width'] * image_observations['image_height']))
        image_observations['size'] = np.where(image_observations['size'] > 1, 1, image_observations[
            'size'])  # There are 3 images with sizes greater than 1, presumably in error, so cap these
        image_observations['x_center'] = image_observations['xmin'] + image_observations['width'] / 2
        image_observations['y_center'] = image_observations['ymin'] + image_observations['height'] / 2
        image_observations['x_dist_from_center'] = image_observations['x_center'] - (
                    image_observations['image_width'] / 2)
        image_observations['y_dist_from_center'] = image_observations['y_center'] - (
                    image_observations['image_height'] / 2)
        image_observations['dist_from_center'] = np.sqrt(
            image_observations['x_dist_from_center'] ** 2
            + image_observations['y_dist_from_center'] ** 2)
        image_observations['distance'] = image_observations['dist_from_center'] / np.sqrt(
            (image_observations['image_width'] / 2) ** 2 + (image_observations['image_height']) ** 2)
        image_observations = image_observations.drop(['xmin', 'ymin', 'width', 'height', 'area',
                                                      'x_center', 'y_center', 'x_dist_from_center',
                                                      'image_width', 'image_height',
                                                      'y_dist_from_center', 'dist_from_center'],
                                                     axis=1)
    image_observations = image_observations.drop_duplicates().reset_index(drop=True)

    category_mapper = {c['id']: c['name'].lower().replace(' ', '_')
                       for c in categories}
    category_mapper[13] = 'street_sign'  # originally "stop_sign"
    category_mapper[37] = 'ball'  # originally "sports_ball"
    category_mapper[46] = 'wineglass'  # originally "wine_glass"
    category_mapper[64] = 'pot_plant'  # originally "potted_plant"
    category_mapper = {k: most_common_synset(v, 'n').name() for k, v in category_mapper.items()}

    image_observations['synset'] = image_observations['category_id'].apply(lambda x: category_mapper[x])
    image_observations = image_observations.drop('category_id', axis=1).reset_index(drop=True)

    return image_observations


def load_visual_genome_labels(include_size_and_location=False):
    """
    Load human annotations of attributes, nodes, and relations that were provided as apart of the Visual Genome dataset.

    These are stored as a json originally, but are converted to csv for easier processing.

    :param include_size_and_location:
    :return:
    """
    relations_path = os.path.join(f'processed_data', f'visual_genome', f'relations.csv')
    nodes_path = os.path.join(f'processed_data', f'visual_genome', f'objects.csv')
    attributes_path = os.path.join(f'processed_data', f'visual_genome', f'attributes.csv')

    if os.path.exists(relations_path) and os.path.exists(nodes_path) and os.path.exists(attributes_path):
        relationships = pd.read_csv(relations_path, converters={'target_synset': literal_eval,
                                                                'source_synset': literal_eval,
                                                                'relationship_synset': literal_eval})
        objects = pd.read_csv(nodes_path, converters={'synset': literal_eval, 'names': literal_eval})
        attributes = pd.read_csv(attributes_path, converters={'names': literal_eval})


    else:

        image_data = json.load(open(os.path.join('input_data', 'visual_genome', 'image_data.json')))
        height_mapper = {i['coco_id']: i['height'] for i in image_data}
        width_mapper = {i['coco_id']: i['width'] for i in image_data}

        coco_to_vg_mapping_info = json.load(open(os.path.join('input_data', 'visual_genome', 'image_data.json')))
        vg_to_coco_mapping_info = {i['image_id']: i['coco_id'] for i in coco_to_vg_mapping_info if
                                   i['coco_id'] is not None}

        # Load from disk
        vg_sg = json.load(
            open('/Users/is28/Documents/Code/bias_propagation/input_data/visual_genome/scene_graphs.json'))

        # Get coco annotations, so we can filter down
        coco_annotations = load_coco_annotations()

        image_observations = {i: [] for i in ['object', 'attribute', 'relationship']}

        # Iterate through images in visual genome
        for img in vg_sg:
            # Check if image is in coco
            if (img['image_id'] in vg_to_coco_mapping_info.keys()
                    and vg_to_coco_mapping_info[img['image_id']] in coco_annotations['image_id'].values):
                # Iterate through objects
                for o in img['objects']:
                    image_observations['object'].append({'image_id': vg_to_coco_mapping_info[img['image_id']],
                                                         'synset': o['synsets'],
                                                         'object_id': o['object_id'],
                                                         'names': o['names']
                                                         })
                    # Unload location information
                    for additional_col in ['h', 'w', 'y', 'x']:
                        if additional_col in o.keys():
                            image_observations['object'][-1][additional_col] = o[additional_col]
                    # Unload attributes, if they are present
                    if 'attributes' in o.keys():
                        for a in o['attributes']:
                            attr_synset = most_common_synset(a, 'a')
                            if attr_synset is not None:
                                attr_synset = attr_synset.name()
                            else:
                                attr_synset = np.nan
                            image_observations['attribute'].append(
                                {'image_id': vg_to_coco_mapping_info[img['image_id']],
                                 'object_id': o['object_id'],
                                 'attribute_id': a,
                                 'names': o['names'],
                                 'synset': attr_synset})
                # Unload relationships in image
                for rel in img['relationships']:
                    source_synset = [o['synsets'] for o in img['objects'] if o['object_id'] == rel['subject_id']][0]
                    target_synset = [o['synsets'] for o in img['objects'] if o['object_id'] == rel['object_id']][0]
                    relationship_sysnet = rel['synsets']
                    image_observations['relationship'].append({'image_id': vg_to_coco_mapping_info[img['image_id']],
                                                               'source_synset': source_synset,
                                                               'source_id': rel['subject_id'],
                                                               'target_synset': target_synset,
                                                               'target_id': rel['object_id'],
                                                               'relationship_synset': relationship_sysnet,
                                                               'relationship_predicate': rel['predicate'],
                                                               'relationship_id': rel['relationship_id']})

        objects = pd.DataFrame(image_observations['object']).sort_values('image_id').reset_index(drop=True)

        # Calculate size and distance from center
        objects['image_width'] = objects['image_id'].map(width_mapper)
        objects['image_height'] = objects['image_id'].map(height_mapper)
        objects['size'] = (objects['w'] * objects['h'] /
                           (objects['image_width'] * objects['image_height']))
        objects['size'] = np.where(objects['size'] > 1, 1, objects[
            'size'])  # Cap images with size greater than 1 (presumably errors)
        objects['x_center'] = objects['x'] + objects['w'] / 2
        objects['y_center'] = objects['y'] + objects['h'] / 2
        objects['x_dist_from_center'] = objects['x_center'] - (objects['image_width'] / 2)
        objects['y_dist_from_center'] = objects['y_center'] - (objects['image_height'] / 2)
        objects['dist_from_center'] = np.sqrt(
            objects['x_dist_from_center'] ** 2
            + objects['y_dist_from_center'] ** 2)
        objects['distance'] = objects['dist_from_center'] / np.sqrt(
            (objects['image_width'] / 2) ** 2 + (objects['image_height']) ** 2)
        objects['distance'] = np.where(objects['distance'] > 1, 1, objects[
            'distance'])  # Cap images with size greater than 1 (presumably errors)
        objects = objects.drop(['x', 'y', 'w', 'h',
                                'x_center', 'y_center', 'x_dist_from_center',
                                'image_width', 'image_height',
                                'y_dist_from_center', 'dist_from_center'],
                               axis=1)

        attributes = pd.DataFrame(image_observations['attribute']).sort_values('image_id').reset_index(drop=True)
        relationships = pd.DataFrame(image_observations['relationship']).sort_values('image_id').reset_index(drop=True)

        os.makedirs(os.path.join(f'processed_data', f'visual_genome'), exist_ok=True)
        objects.to_csv(nodes_path, index=False)
        attributes.to_csv(attributes_path, index=False)
        relationships.to_csv(relations_path, index=False)

    if not include_size_and_location:
        objects = objects.drop(['size', 'distance'], axis=1)
    return objects, relationships, attributes


def load_zhao_image_paths():
    """
    Load paths for images that have been annotated for demographics by Zhao et al.

    """
    image_annotations = pd.read_csv(os.path.join('input_data', 'COCO 2014 Val Demographic Annotations',
                                                 'images_val2014.csv'))
    image_paths = [os.path.join('input_data', 'val2014', f'COCO_val2014_{int(id):012d}.jpg')
                   for id in image_annotations['id'].values]
    image_paths = pd.DataFrame({'image_id': image_annotations['id'].values, 'path': image_paths})
    return image_paths


if __name__ == '__main__':
    load_visual_genome_labels()
