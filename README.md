# High-order-IE


This repo contains the code used for the high-order join information extraction in Jia et al. (2023), [Modeling Instance Interactions for Joint Information Extraction with Neural High-Order Conditional Random Field](https://aclanthology.org/2023.acl-long.766/#). The codebase of this repo is extended from OneIE v0.4.8

# Requirements

Python 3.7
Python packages
- PyTorch 1.0+ (Install the CPU version if you use this tool on a machine without GPUs)
- transformers 3.0.2 (It seems using transformers 3.1+ may cause some model loading issue)
- tqdm
- lxml
- nltk


# How to Run

## Pre-processing
The data pre-processing follows OneIE.

### DyGIE++ to OneIE format
The `prepreocessing/process_dygiepp.py` script converts datasets in DyGIE++
format (https://github.com/dwadden/dygiepp/tree/master/scripts/data/ace-event) to
the format used by OneIE. Example:

```
python preprocessing/process_dygiepp.py -i train.json -o train.oneie.json
```

Arguments:
- -i, --input: Path to the input file.
- -o, --output: Path to the output file.

### ACE2005 to OneIE format
The `prepreocessing/process_ace.py` script converts raw ACE2005 datasets to the
format used by OneIE. Example:

```
python preprocessing/process_ace.py -i <INPUT_DIR>/LDC2006T06/data -o <OUTPUT_DIR>
  -s resource/splits/ACE05-E -b bert-large-cased -c <BERT_CACHE_DIR> -l english
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your LDC2006T06
  package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -s, --split: Path to the split directory. We provide document id lists for all
  datasets used in our paper in `resource/splits`.
- -l, --lang: Language (options: english, chinese).

### ERE to OneIE format
The `prepreocessing/process_ere.py` script converts raw ERE datasets (LDC2015E29,
LDC2015E68, LDC2015E78, LDC2015E107) to the format used by OneIE. 

```
python preprocessing/process_ere.py -i <INPUT_DIR>/data -o <OUTPUT_DIR>
  -b bert-large-cased -c <BERT_CACHE_DIR> -l english -d normal
```

Arguments:
- -i, --input: Path to the input directory (`data` folder in your ERE package).
- -o, --output: Path to the output directory.
- -b, --bert: Bert model name.
- -c, --bert_cache_dir: Path to the BERT cache directory.
- -d, --dataset: Dataset type: normal, r2v2, parallel, or spanish.
- -l, --lang: Language (options: english, spanish).

This script only supports:
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V1
- LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
- LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
- LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
- LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2



