import json
import os

import numpy as np
import pandas as pd
from nltk.corpus import wordnet, verbnet

from measurement.loading import load_coco_annotations, load_visual_genome_labels, \
    load_scene_graphs_in_model_captions
from measurement.utils import most_common_synset


def incorrect_words(model_name, dataset, heuristic):
    """
    Find the words that are incorrectly included in captions, according to the criteria provided in paper
    Filters down to only include words that are in the heuristic

    :param model_name:
    :param dataset:
    :param heuristic:
    :return:
    """
    if dataset == 'coco':
        # "we treated stage 1 (human-generated
        # labels) as “ground truth,” thereby assuming the human-generated la-
        # bels are high quality (e.g., if an object is labeled as fruit and not apple,
        # we assume this is because the object is not identifiable as anything
        # more specific than a fruit)."
        ni_obj, ni_rel, ni_attr = load_nonimageable_synsets()
        gt_objs, gt_rels, gt_attrs = load_visual_genome_labels()
        pred_objs, pred_rels, pred_attrs = load_scene_graphs_in_model_captions(model_name, dataset)
    else:
        raise NotImplementedError

    pred_objs['ni_obj'] = pred_objs['synset'].isin(ni_obj) & pred_objs['synset'].isin(heuristic.index)
    pred_rels['ni_rel'] = pred_rels['synset'].isin(ni_rel) & pred_objs['synset'].isin(heuristic.index)
    pred_attrs['ni_attr'] = pred_attrs['synset'].isin(ni_attr) & pred_objs['synset'].isin(heuristic.index)

    spec_and_hal_objs = []
    spec_and_hal_attrs = []
    spec_and_hal_rels = []
    images_in_both = set(gt_objs['image_id'].unique()).intersection(set(pred_objs['image_id'].unique()))
    for img_id in list(
            images_in_both):  # Loop through all images that appear in both the human generated labels and the model generated labels
        gt_objs_in_img = gt_objs[gt_objs['image_id'] == img_id]  # Get all predicted objects in the image
        pred_objs_in_img = pred_objs[pred_objs['image_id'] == img_id]  # Get all ground truth objects in the image

        obj_matches = {}  # Dictionary to store the matches between the predicted objects and the ground truth objects
        for _, pred_obj in pred_objs_in_img.iterrows():  # Loop through all predicted objects in the image
            if pred_obj['synset'] is np.nan:  # Some objects do not have synsets, skip these
                continue
            pred_syn = wordnet.synset(pred_obj['synset'])  # Get the synset of the predicted object

            # Find the corresponding object in the human generated labels
            best_sim = -np.Inf
            for _, gt_obj in gt_objs_in_img.iterrows():
                if type(gt_obj['synset']) is not list:  # Some objects have multiple synsets, so loop through these
                    gt_obj['synset'] = [gt_obj['synset']]
                for gt_syn in gt_obj['synset']:
                    if gt_syn is np.nan:  # Skip if the synset is nan
                        continue
                    gt_syn = wordnet.synset(gt_syn)
                    if gt_syn.pos() != pred_syn.pos():  # Make sure the POS is the same
                        continue
                    sim = pred_syn.lch_similarity(gt_syn)  # Use Leacock Chodorow Similarity to compare the synsets
                    if sim > best_sim:
                        best_sim = sim
                        best_gt_syn = gt_syn
                        best_gt_id = gt_obj['object_id']
                        best_gt_name = ', '.join(gt_obj['names'])
                obj_matches[pred_obj['object_id']] = (best_gt_id, best_gt_name, best_gt_syn)

            # Find whether object synset is too specific
            pred_hypernym_list = [val for sublist in pred_syn.hypernym_paths() for val in sublist]
            gt_hypernym_list = [val for sublist in best_gt_syn.hypernym_paths() for val in sublist]
            too_specific = (pred_syn != best_gt_syn and best_gt_syn in pred_hypernym_list) and pred_syn.name() in heuristic.index
            hallucination = (pred_syn != best_gt_syn and pred_syn not in gt_hypernym_list and best_gt_syn not in pred_hypernym_list) and pred_syn.name() in heuristic.index
            spec_and_hal_objs.append({'image_id': img_id,
                                      'object_id': pred_obj['object_id'], 'gt_obj_id': best_gt_id,
                                      'synset': pred_syn.name(), 'gt_obj_syn': best_gt_syn.name(),
                                      'node_name': pred_obj['node_name'], 'gt_obj_syn_name': best_gt_name,
                                      'too_specific': too_specific,
                                      'hallucination': hallucination,
                                      })

            # Find attributes that are too specific
            pred_attr_for_obj = pred_attrs[
                (pred_attrs['image_id'] == img_id) & (pred_attrs['object_id'] == pred_obj['object_id'])]
            gt_attr_for_obj = gt_attrs[(gt_attrs['image_id'] == img_id) & (gt_attrs['object_id'] == best_gt_id)]
            for _, pred_attr in pred_attr_for_obj.iterrows():
                if pred_attr['synset'] is np.nan:
                    continue
                pred_attr_syn = wordnet.synset(pred_attr['synset'])
                too_specific = False
                too_general = False
                gt_attr_syns = []
                gt_attr_syn_names = []
                complete_gt_hypernym_list = []
                for _, gt_attr in gt_attr_for_obj.iterrows():
                    if gt_attr['synset'] is np.nan:
                        continue
                    gt_attr_syn = wordnet.synset(gt_attr['synset'])
                    gt_attr_syns += [gt_attr_syn.name()]
                    gt_attr_syn_names += [gt_attr['attribute_id']]
                    pred_hypernym_list = [val for sublist in pred_attr_syn.hypernym_paths() for val in sublist]
                    gt_hypernym_list = [val for sublist in gt_attr_syn.hypernym_paths() for val in sublist]
                    complete_gt_hypernym_list += gt_hypernym_list
                    if (
                            pred_attr_syn != gt_attr_syn and gt_attr_syn in pred_hypernym_list) and pred_attr_syn.name() in heuristic.index:
                        too_specific = True
                    if (pred_attr_syn != gt_attr_syn and pred_attr_syn in gt_hypernym_list):
                        too_general = True

                hallucination = (
                                            pred_attr_syn.name() not in gt_attr_syns and pred_attr_syn not in complete_gt_hypernym_list and not too_general) and pred_attr_syn.name() in heuristic.index
                spec_and_hal_attrs.append({'image_id': img_id,
                                           'object_id': pred_obj['object_id'], 'gt_object_id': best_gt_id,
                                           'pred_object_synset': pred_syn.name(),
                                           'gt_object_synset': best_gt_syn.name(),
                                           'pred_object_syn_name': pred_obj['node_name'],
                                           'gt_object_syn_name': best_gt_name,
                                           'synset': pred_attr_syn.name(), 'gt_attr_syn': ', '.join(gt_attr_syns),
                                           'attribute': pred_attr['attribute'],
                                           'gt_attr_syn_name': ', '.join(gt_attr_syn_names),
                                           'too_specific': too_specific,
                                           'hallucination': hallucination
                                           })

        # Find relations that are too specific
        for pred_obj_id, (gt_obj_id, gt_obj_name, gt_obj_syn) in obj_matches.items():

            pred_rel_for_obj = pred_rels[(pred_rels['image_id'] == img_id) & (pred_rels['object_id'] == pred_obj_id)]
            gt_rel_for_obj = gt_rels[(gt_rels['image_id'] == img_id) & (gt_rels['source_id'] == gt_obj_id)]
            for _, pred_rel in pred_rel_for_obj.iterrows():
                if pred_rel['synset'] is np.nan or pred_rel['target'] not in obj_matches.keys():
                    continue

                pred_obj = pred_objs_in_img[pred_objs_in_img['object_id'] == pred_obj_id].iloc[0]
                pred_target_obj = pred_objs_in_img[pred_objs_in_img['object_id'] == pred_rel['target']].iloc[0]
                gt_target_syn = obj_matches[pred_rel['target']][2]
                gt_target_syn_name = obj_matches[pred_rel['target']][1]

                pred_rel_syn = wordnet.synset(pred_rel['synset'])
                pred_hypernym_list = [val for sublist in pred_rel_syn.hypernym_paths() for val in sublist]

                # Only look at relations that refer to same target
                gt_rel_with_same_target = gt_rel_for_obj[
                    (gt_rel_for_obj['target_id'] == obj_matches[pred_rel['target']][0])]
                too_specific = False
                too_general = False
                gt_rel_syns = []
                gt_rel_syn_names = []
                gt_rel_relationship_ids = []
                complete_gt_hypernym_list = []
                for _, gt_rel in gt_rel_with_same_target.iterrows():
                    if gt_rel['relationship_synset'] is np.nan:  # Make sure the synset is not nan
                        continue
                    for gt_rel_syn in gt_rel['relationship_synset']:
                        gt_rel_syn = wordnet.synset(gt_rel_syn)
                        gt_rel_syns += [gt_rel_syn.name()]
                        gt_rel_syn_names += [gt_rel['relationship_predicate']]
                        gt_rel_relationship_ids += [gt_rel['relationship_id']]
                        gt_hypernym_list = [val for sublist in gt_rel_syn.hypernym_paths() for val in sublist]
                        complete_gt_hypernym_list += gt_hypernym_list
                        if (
                                pred_rel_syn != gt_rel_syn and gt_rel_syn in pred_hypernym_list) and pred_rel_syn.name() in heuristic.index:
                            too_specific = True
                        if pred_rel_syn != gt_rel_syn and pred_rel_syn in gt_hypernym_list:
                            too_general = True

                hallucination = (pred_rel_syn.name() not in gt_rel_syns and pred_rel_syn not in complete_gt_hypernym_list and not too_general) and pred_rel_syn.name() in heuristic.index

                spec_and_hal_rels.append({'image_id': img_id,
                                          'object_id': pred_obj_id, 'gt_obj_id': gt_obj_id,
                                          'pred_obj_syn': pred_obj['synset'], 'gt_obj_syn': gt_obj_syn.name(),
                                          'pred_obj_syn_name': pred_obj['node_name'], 'gt_obj_syn_name': gt_obj_name,

                                          'target': pred_rel['target'],
                                          'gt_target_id': obj_matches[pred_rel['target']][0],
                                          'pred_target_syn': pred_target_obj['synset'],
                                          'gt_target_syn': gt_target_syn.name(),
                                          'pred_target_syn_name': pred_target_obj['node_name'],
                                          'gt_target_syn_name': gt_target_syn_name,

                                          'synset': pred_rel_syn.name(), 'gt_rel_syn': gt_rel_syns,
                                          'reln': pred_rel['reln'], 'gt_rel_syn_name': gt_rel_syn_names,
                                          'gt_rel_id': gt_rel_relationship_ids,
                                          'too_specific': too_specific,
                                          'hallucination': hallucination
                                          })

    pred_objs = pred_objs.merge(pd.DataFrame(spec_and_hal_objs),
                                on=['object_id', 'image_id', 'node_name', 'synset'])

    pred_attrs = pred_attrs.merge(pd.DataFrame(spec_and_hal_attrs), on=['attribute', 'image_id', 'synset', 'object_id'])

    pred_rels = pred_rels.merge(pd.DataFrame(spec_and_hal_rels),
                                on=['object_id', 'synset', 'target', 'image_id', 'reln'])

    objs_by_id = pred_objs.groupby('image_id')[['ni_obj', 'too_specific', 'hallucination']].max().rename(
        columns={'too_specific': 'ts_obj', 'hallucination': 'h_obj'})
    attrs_by_id = pred_attrs.groupby('image_id')[['ni_attr', 'too_specific', 'hallucination']].max().rename(
        columns={'too_specific': 'ts_attr', 'hallucination': 'h_attr'})
    rels_by_id = pred_rels.groupby('image_id')[['ni_rel', 'too_specific', 'hallucination']].max().rename(
        columns={'too_specific': 'ts_rel', 'hallucination': 'h_rel'})

    together = pd.concat([objs_by_id, attrs_by_id, rels_by_id], axis=1)
    together['any'] = together.any(axis=1)
    together['ni'] = together['ni_obj'] | together['ni_attr'] | together['ni_rel']
    together['ts'] = together['ts_obj'] | together['ts_attr'] | together['ts_rel']
    together['h'] = together['h_obj'] | together['h_attr'] | together['h_rel']

    return together


def filter_by_heuristic(threshold=0.005):
    """
    Load the words that the heuristic determines are more likely to be stereotypical

    :param threshold:
    :return:
    """

    # TODO: UPDATE
    # Calculate p(synset)

    # "For COCO, the human-generated labels come from a union of annotations from COCO [44] and Visual Genome"
    coco_image_observations = load_coco_annotations()
    vg_image_observations = list(load_visual_genome_labels())

    # Unload synset lists from vg
    synset_column_names = ['synset', 'relationship_synset']
    for j, synset_col in enumerate(synset_column_names):
        i = 0
        all_na = False
        while not all_na:
            vg_image_observations[j][f'syn_{i}'] = vg_image_observations[j][synset_col].str[i]
            all_na = vg_image_observations[j][f'syn_{i}'].isna().all()
            i += 1
        vg_image_observations[j] = vg_image_observations[j].melt(id_vars=['image_id'],
                                                                 value_vars=[f'syn_{k}' for k in range(i)])
        vg_image_observations[j] = vg_image_observations[j].drop(columns='variable')
        vg_image_observations[j] = vg_image_observations[j][~vg_image_observations[j]['value'].isna()]
        vg_image_observations[j] = vg_image_observations[j].rename(columns={'value': 'synset'})

    image_observations = pd.concat(
        [coco_image_observations] +
        [vg_image_observations[i][['image_id', 'synset']] for i in range(len(vg_image_observations))]
    ).drop_duplicates()

    wang_image_annotations = pd.read_csv(os.path.join('input_data', 'COCO 2014 Val Demographic Annotations',
                                                      'images_val2014.csv')).rename(columns={'id': 'image_id'})
    image_observations = image_observations.merge(wang_image_annotations[['image_id', 'bb_skin', 'bb_gender']],
                                                  left_on='image_id', right_on='image_id', how='left')


    marginal_synset_probabilities = (
            image_observations['synset'].value_counts()
            / len(image_observations['image_id'].unique())
    )

    # calculate p(group)
    marginal_probabilities = wang_image_annotations[['bb_skin', 'bb_gender']].value_counts() / len(
        wang_image_annotations)
    marginal_skin_probabilities = wang_image_annotations['bb_skin'].value_counts() / len(wang_image_annotations)
    marginal_gender_probabilities = wang_image_annotations['bb_gender'].value_counts() / len(wang_image_annotations)

    # Calculate p(group) * p(synset)
    marg = marginal_probabilities.to_numpy().reshape((-1, 1)) @ marginal_synset_probabilities.to_numpy().reshape(
        (1, -1))
    marg = pd.DataFrame(marg, index=marginal_probabilities.index,
                        columns=marginal_synset_probabilities.index).sort_index()
    skin_marg = marginal_skin_probabilities.to_numpy().reshape(
        (-1, 1)) @ marginal_synset_probabilities.to_numpy().reshape((1, -1))
    skin_marg = pd.DataFrame(skin_marg, index=marginal_skin_probabilities.index,
                             columns=marginal_synset_probabilities.index).sort_index()
    gender_marg = marginal_gender_probabilities.to_numpy().reshape(
        (-1, 1)) @ marginal_synset_probabilities.to_numpy().reshape((1, -1))
    gender_marg = pd.DataFrame(gender_marg, index=marginal_gender_probabilities.index,
                               columns=marginal_synset_probabilities.index).sort_index()

    # Calculate p(group, synset) of each synset
    skin_joint = (image_observations[['synset', 'bb_skin']].value_counts() / len(wang_image_annotations)).reset_index()
    skin_joint = skin_joint.pivot(index='bb_skin', columns='synset', values=0).fillna(0).sort_index()
    skin_marg = skin_marg[skin_joint.columns]

    gender_joint = (
                image_observations[['synset', 'bb_gender']].value_counts() / len(wang_image_annotations)).reset_index()
    gender_joint = gender_joint.pivot(index='bb_gender', columns='synset', values=0).fillna(0).sort_index()
    gender_marg = gender_marg[gender_joint.columns]

    joint = (image_observations[['synset', 'bb_skin', 'bb_gender']].value_counts() / len(
        wang_image_annotations)).reset_index()
    joint = joint.pivot(index=['bb_skin', 'bb_gender'], columns='synset', values=0).fillna(0).sort_index()
    marg = marg[joint.columns]

    synset_heuristic = pd.concat([
        skin_joint - skin_marg,
        gender_joint - gender_marg
    ]).max()

    synset_heuristic = synset_heuristic[synset_heuristic > threshold].sort_values(ascending=False)
    return synset_heuristic


def load_nonimageable_synsets():
    # Todo: Make sure replacing terms like '?climb' with 'climb' is correct
    # "For objects, we used the non-imageable synsets in the people
    # # subtree of WordNet [77]"
    imageability_info_objects = pd.read_csv(os.path.join('input_data', 'imageability_scores.txt'),
                                            header=None, sep='\s', engine='python', comment='#')
    imageability_info_objects.columns = ['synset', 'imageability']
    non_imageable_objects = imageability_info_objects[
        imageability_info_objects['imageability'] < 4].copy()  # Using cutoff defined by Yang et al.
    non_imageable_objects['pos'] = non_imageable_objects['synset'].str.extract(r'([a-z]+)')
    non_imageable_objects['offset'] = non_imageable_objects['synset'].str.extract(r'([0-9]+)')
    obj_synsets = []
    for i, row in non_imageable_objects.iterrows():
        obj_synsets.append(wordnet.synset_from_pos_and_offset(row['pos'], int(row['offset'])).name())

    # "for attributes, we used those adjectives in
    # a list of people-descriptor categories [70] that we determined to be
    # non-imageable (i.e., attractiveness, ethnicity, judgment, mood, occu-
    # pation or social group, relation, and state)"
    # They don't state whether they use flickr or visualgenome, so I assume they use both
    categories = ['attractiveness', 'ethnicity', 'judgment', 'mood', 'occupation-or-social-group', 'relation', 'state']
    non_imageable_attributes = []
    for dataset in ['Flickr30k', 'VisualGenome']:
        for cat in categories:
            save_path = os.path.join('input_data', 'LabelingPeople', dataset,
                                     'resources', 'categories', f'{cat}.txt')
            if os.path.exists(save_path):
                attributes = pd.read_csv(save_path, header=None, comment='#')
                attributes.columns = ['attribute']
                attributes['dataset'] = dataset
                attributes['category'] = cat
                non_imageable_attributes.append(attributes)
    non_imageable_attributes = pd.concat(non_imageable_attributes)
    attr_synsets = [most_common_synset(a, 'a').name() for a in non_imageable_attributes['attribute'] if
                    most_common_synset(a, 'a') is not None]
    attr_synsets = np.unique(attr_synsets).tolist()

    # "for relationships, we used any verb not included in Visual VerbNet or in {have, in}."
    vvn = json.load(open(os.path.join('input_data', 'visual_verbnet_beta2015.json')))
    imageable_verbs = [v['name'].replace('_', ' ').lower().replace('?', '') for v in vvn['visual_actions']] + ['have',
                                                                                                               'in']
    all_vn_classes = verbnet.classids()
    all_vn_lemmas = []
    while len(all_vn_classes) > 0:
        vn_class = all_vn_classes.pop()
        all_vn_lemmas += verbnet.lemmas(vn_class)
        if len(verbnet.subclasses(vn_class)) > 0:
            all_vn_classes += verbnet.subclasses(vn_class)
    all_vn_lemmas = np.unique(all_vn_lemmas)
    rel_synsets = [v.replace('_', ' ').lower().replace('?', '') for v in all_vn_lemmas
                   if v not in imageable_verbs]
    rel_synsets = [most_common_synset(a, 'v').name() for a in rel_synsets if most_common_synset(a, 'v') is not None]

    return obj_synsets, rel_synsets, attr_synsets


if __name__ == '__main__':
    heur = filter_by_heuristic()
    save_dir = os.path.join('processed_data','measurements','incorrect_word')
    os.makedirs(save_dir, exist_ok=True)
    for model_name in ['clipcap_coco_mlp', 'clipcap_coco_transformer','clipcap_conceptual_mlp']:
        words = incorrect_words(model_name=model_name, dataset='coco', heuristic=heur)
        words.to_csv(os.path.join(save_dir, f'{model_name}.csv'))
