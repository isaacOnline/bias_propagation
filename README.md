# Measuring Representational Harms in Image Captioning
This repository contains unofficial implementations for some of the harm metrics introduced by Wang et al. in 
[Measuring Representational Harms in Image Captioning](https://doi.org/10.1145/3531146.3533099). It implements the 
`stereotyping` measures, as well as one of the `demeaning` measures. 

These measures can be run using the Python scripts provided in the `measurement` directory. The script 
`incorrect_word.py` can be used for the measurements described in section 4.1.1 of the paper. This script will create a 
file in `processed_data/measurements/incorrect_word`, detailing, for each caption, whether it contains a non-imageable concept, an 
overly specific concept, or a hallucinated concept. 
The script `correct_word.py`can be used for the measurements described in section 4.1.2. The script will create a file
in `processed_data/measurements/correct_word` containing th logistic regressions models fit as described in the section,
as well as accuracy from those models
The script `demeaning_words.py` can be used for the measurements described in section 4.2.1. The script will create
a file in `processed_data/measurements/demeaning` detailing, for each image whether it contains demeaning words (based 
on the three criteria described in the section), and if so, which words meet each criterion.

The scripts will require captions to be run, and so the repository also contains code for generating captions using 
[ClipCap](https://github.com/rmokady/CLIP_prefix_caption). Once the weights have been downloaded, 
`caption_generation/clipcap.py` will generate captions. Alternatively, captions for images can be stored in 
`processed_data/{dataset_name}_captions`, under the name of the model used for generating them, e.g. 
`processed_data/coco_captions/clipcap_conceptual_mlp.csv`. This csv should have the columns `image_id` and `caption`.

## Accessing Data:
As the data used in this project is not my own, I have provided links below to where I got the data from.
1. Download race annotations from [Wang et al.](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Understanding_and_Evaluating_Racial_Biases_in_Image_Captioning_ICCV_2021_paper.html). As of writing, they can be requested [here](https://princetonvisualai.github.io/imagecaptioning-bias/). The folder `COCO 2014 Val Demographic Annotations` should be placed in the `input_data` directory. 
2. Download the [COCO](https://cocodataset.org/#download) datasets (val2014 images and val2014 annotations). Both the image folder (`val2014`) and the annotations folder (`annotations`) should be placed in the `input_data` directory.
3. Download the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) dataset. Associated files should be placed in `input_data/visual_genome`.
4. Download imageability info from [Yang et al.](https://doi.org/10.1145/3351095.3375709) [here](https://image-net.org/filtering-and-balancing/imageability_scores.txt). The file should be placed in the `input_data` directory. 
5. Download unsafe synset info from [Yang et al.](https://doi.org/10.1145/3351095.3375709) [here](https://www.image-net.org/filtering-and-balancing/unsafe_synsets.txt). The file should be placed in the `input_data` directory.
6. Download Visual VerbNet from [here](http://www.vision.caltech.edu/~mronchi/projects/Cocoa/#download).  I used the beta, as the 1.0 version was not available as I was working on this project. I've reached out to multiple authors of the paper about accessing the dataset, but haven't heard back. My code refers to the file `visual_verbnet_beta2015.json`, which you will need to change if you get access to the original version.
7. Download harmful attribute data from [van Miltenburg et al.](http://dx.doi.org/10.18653/v1/W18-6550) [here](https://github.com/evanmiltenburg/LabelingPeople). The `LabelingPeople` folder should be placed in the `input_data` directory.
8. Download the Conceptual Captions validation set [here](https://ai.google.com/research/ConceptualCaptions/download). The file should be saved in a directory called `input_data/conceptual_captions`.
9. If captions are to be generated using ClipCap, download pre-trained weights [here](https://github.com/rmokady/CLIP_prefix_caption). These weights should be saved in `references/CLIP_prefix_caption/pretrained_models`.

## Other Requirements:
The conda environment for this project is detailed in `environment.yml`. You will also need to download and install 
[Stanford scene graph parser 3.6.0](https://nlp.stanford.edu/software/scenegraph-parser.shtml) to use the project. The directory `stanford-corenlp-full-2015-12-09` should
also be placed in the `references directory`.
