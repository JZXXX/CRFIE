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


## Training
- `cd` to the root directory of this package
- Set the environment variable PYTHONPATH to the current directory.
  For example, if you unpack this package to `~/High-order-IE`, run: `export PYTHONPATH=~/High-order-IE`
  
Because our framework is a pipeline schema, you should first train the Node Identification model and save the checkpoint in a directory.
- Run the command line to train an identification model: `python train_ident.py -c <CONFIG_FILE_PATH>`.

Then train the high-order classification model.
- `python train.py -c <CONFIG_FILE_PATH>`.
- One example configuration file is in `config/baseline.json`. Fill in the following paths in the configuration file:
  - BERT_CACHE_DIR: Pre-trained BERT models, configs, and tokenizers will be downloaded to this directory.
  - TRAIN_FILE_PATH, DEV_FILE_PATH, TEST_FILE_PATH: Path to the training/dev/test/files.
  - OUTPUT_DIR: The model will be saved to subfolders in this directory.
  - VALID_PATTERN_DIR: Valid patterns created based on the annotation guidelines or training set. Example files are provided in `resource/valid_patterns`.
  - Set NER_SCORE and SPLIT_TRAIN to be true: Our base pipeline model with different scoring functions of OneIE.
  - IDENT_MODEL_PATH: Path to a checkpoint of the saved node identification model.
  The following hyper-parameters control the high-order part:
  - USE_*: Whether to use the corresponding high-order factor.
  - DECOMP_SIZE and MFVI_ITER: Hyperparameters of mean field variational inference (refer to paper).

## Evaluation
Example command line to test a file input: `python predict.py -m <best.role.mdl> -i <input_dir> -o <output_dir> --format json`
  + Arguments:
    - -m, --model_path: Path to the trained model.
    - -i, --input_dir: Path to the input directory. LTF format sample files can be found in the `input` directory.
    - -o, --output_dir: Path to the output directory (json format). Output files are in the JSON format. Sample files can be found in the `output` directory.
    - --gpu: (optional) Use GPU
    - -d, --device: (optional) GPU device index (for multi-GPU machines).
    - -b, --batch_size: (optional) Batch size. For a 16GB GPU, a batch size of 10~15 is a reasonable value.
    - --max_len: (optional) Max sentence length. Sentences longer than this value will be ignored. You may need to decrease `batch_size` if you set `max_len` to a larger number.
    - --lang: (optional) Model language.
    - --format: Input file format (txt, ltf, or json).

# Data Format

Processed input example:
```
{"doc_id": "AFP_ENG_20030401.0476", "sent_id": "AFP_ENG_20030401.0476-5", "entity_mentions": [{"id": "AFP_ENG_20030401.0476-5-E0", "start": 0, "end": 1, "entity_type": "GPE", "mention_type": "UNK", "text": "British"}, {"id": "AFP_ENG_20030401.0476-5-E1", "start": 1, "end": 2, "entity_type": "PER", "mention_type": "UNK", "text": "Chancellor"}, {"id": "AFP_ENG_20030401.0476-5-E2", "start": 4, "end": 5, "entity_type": "ORG", "mention_type": "UNK", "text": "Exchequer"}, {"id": "AFP_ENG_20030401.0476-5-E3", "start": 5, "end": 7, "entity_type": "PER", "mention_type": "UNK", "text": "Gordon Brown"}, {"id": "AFP_ENG_20030401.0476-5-E4", "start": 12, "end": 13, "entity_type": "PER", "mention_type": "UNK", "text": "head"}, {"id": "AFP_ENG_20030401.0476-5-E5", "start": 15, "end": 16, "entity_type": "GPE", "mention_type": "UNK", "text": "country"}, {"id": "AFP_ENG_20030401.0476-5-E6", "start": 18, "end": 19, "entity_type": "ORG", "mention_type": "UNK", "text": "regulator"}, {"id": "AFP_ENG_20030401.0476-5-E7", "start": 22, "end": 23, "entity_type": "PER", "mention_type": "UNK", "text": "chairman"}, {"id": "AFP_ENG_20030401.0476-5-E8", "start": 25, "end": 26, "entity_type": "ORG", "mention_type": "UNK", "text": "watchdog"}, {"id": "AFP_ENG_20030401.0476-5-E9", "start": 27, "end": 30, "entity_type": "ORG", "mention_type": "UNK", "text": "Financial Services Authority"}], "relation_mentions": [{"relation_type": "ORG-AFF", "id": "AFP_ENG_20030401.0476-5-R0", "arguments": [{"entity_id": "AFP_ENG_20030401.0476-5-E1", "text": "Chancellor", "role": "Arg-1"}, {"entity_id": "AFP_ENG_20030401.0476-5-E2", "text": "Exchequer", "role": "Arg-2"}]}, {"relation_type": "ORG-AFF", "id": "AFP_ENG_20030401.0476-5-R1", "arguments": [{"entity_id": "AFP_ENG_20030401.0476-5-E4", "text": "head", "role": "Arg-1"}, {"entity_id": "AFP_ENG_20030401.0476-5-E6", "text": "regulator", "role": "Arg-2"}]}, {"relation_type": "ORG-AFF", "id": "AFP_ENG_20030401.0476-5-R2", "arguments": [{"entity_id": "AFP_ENG_20030401.0476-5-E7", "text": "chairman", "role": "Arg-1"}, {"entity_id": "AFP_ENG_20030401.0476-5-E9", "text": "Financial Services Authority", "role": "Arg-2"}]}], "event_mentions": [{"event_type": "Personnel:Nominate", "id": "AFP_ENG_20030401.0476-5-EV0", "trigger": {"start": 9, "end": 10, "text": "named"}, "arguments": [{"entity_id": "AFP_ENG_20030401.0476-5-E4", "text": "head", "role": "Person"}]}], "tokens": ["British", "Chancellor", "of", "the", "Exchequer", "Gordon", "Brown", "on", "Tuesday", "named", "the", "current", "head", "of", "the", "country", "'s", "energy", "regulator", "as", "the", "new", "chairman", "of", "finance", "watchdog", "the", "Financial", "Services", "Authority", "(", "FSA", ")", "."], "pieces": ["British", "Chancellor", "of", "the", "Ex", "##che", "##quer", "Gordon", "Brown", "on", "Tuesday", "named", "the", "current", "head", "of", "the", "country", "'", "s", "energy", "regulator", "as", "the", "new", "chairman", "of", "finance", "watch", "##dog", "the", "Financial", "Services", "Authority", "(", "F", "##SA", ")", "."], "token_lens": [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1], "sentence": "British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country 's energy regulator as the new chairman of finance watchdog the Financial Services Authority ( FSA ) ."}
```

The "start" and "end" of entities and triggers are token indices. The "arguments" of a relation refer to its head entity and tail entity.


Output example:
```
{"doc_id": "HC0003PYD", "sent_id": "HC0003PYD-16", "token_ids": ["HC0003PYD:2295-2296", "HC0003PYD:2298-2304", "HC0003PYD:2305-2305", "HC0003PYD:2307-2311", "HC0003PYD:2313-2318", "HC0003PYD:2320-2325", "HC0003PYD:2327-2329", "HC0003PYD:2331-2334", "HC0003PYD:2336-2337", "HC0003PYD:2339-2348", "HC0003PYD:2350-2351", "HC0003PYD:2353-2360", "HC0003PYD:2362-2362", "HC0003PYD:2364-2367", "HC0003PYD:2369-2376", "HC0003PYD:2378-2383", "HC0003PYD:2385-2386", "HC0003PYD:2388-2390", "HC0003PYD:2392-2397", "HC0003PYD:2399-2401", "HC0003PYD:2403-2408", "HC0003PYD:2410-2412", "HC0003PYD:2414-2415", "HC0003PYD:2417-2425", "HC0003PYD:2427-2428", "HC0003PYD:2430-2432", "HC0003PYD:2434-2437", "HC0003PYD:2439-2441", "HC0003PYD:2443-2447", "HC0003PYD:2449-2450", "HC0003PYD:2452-2454", "HC0003PYD:2456-2464", "HC0003PYD:2466-2472", "HC0003PYD:2474-2480", "HC0003PYD:2481-2481", "HC0003PYD:2483-2485", "HC0003PYD:2487-2491", "HC0003PYD:2493-2502", "HC0003PYD:2504-2509", "HC0003PYD:2511-2514", "HC0003PYD:2516-2523", "HC0003PYD:2524-2524"], "tokens": ["On", "Tuesday", ",", "North", "Korean", "leader", "Kim", "Jong", "Un", "threatened", "to", "detonate", "a", "more", "powerful", "H-bomb", "in", "the", "future", "and", "called", "for", "an", "expansion", "of", "the", "size", "and", "power", "of", "his", "country's", "nuclear", "arsenal", ",", "the", "state", "television", "agency", "KCNA", "reported", "."], "graph": {"entities": [[3, 5, "GPE", "NAM", 1.0], [5, 6, "PER", "NOM", 0.2], [6, 9, "PER", "NAM", 0.5060472888322202], [15, 16, "WEA", "NOM", 0.5332313915378754], [30, 31, "PER", "PRO", 1.0], [32, 33, "WEA", "NOM", 1.0], [33, 34, "WEA", "NOM", 0.5212696155645499], [36, 37, "GPE", "NOM", 0.4998288792916457], [38, 39, "ORG", "NOM", 1.0], [39, 40, "ORG", "NAM", 0.5294904130032032]], "triggers": [[11, 12, "Conflict:Attack", 1.0]], "relations": [[1, 0, "ORG-AFF", 1.0]], "roles": [[0, 2, "Attacker", 0.4597024700555278], [0, 3, "Instrument", 1.0]]}}
```

The output format is the same as OneIE.

OneIE save results in JSON format. Each line is a JSON object for a sentence containing the following fields:
+ doc_id (string): Document ID
+ sent_id (string): Sentence ID
+ tokens (list): A list of tokens
+ token_ids (list): A list of token IDs (doc_id:start_offset-end_offset)
+ graph (object): Information graph predicted by the model
  - entities (list): A list of predicted entities. Each item in the list has exactly
  four values: start_token_index, end_token_index, entity_type, mention_type, score.
  For example, "[3, 5, "GPE", "NAM", 1.0]" means the index of the start token is 3, 
  index of the end token is 4 (5 - 1), entity type is GPE, mention type is NAM,
  and local score is 1.0.
  - triggers (list): A list of predicted triggers. It is similar to `entities`, while
  each item has three values: start_token_index, end_token_index, event_type, score.
  - relations (list): A list of predicted relations. Each item in the list has
  three values: arg1_entity_index, arg2_entity_index, relation_type, score.
  In the following example, `[1, 0, "ORG-AFF", 0.52]` means there is a ORG-AFF relation
  between entity 1 ("leader") and entity 0 ("North Korean") with a local
  score of 0.52.
  The order of arg1 and arg2 can be ignored for "SOC-PER" as this relation is 
  symmetric.
  - roles (list): A list of predicted argument roles. Each item has three values:
  trigger_index, entity_index, role, score.
  In the following example, `[0, 2, "Attacker", 0.8]` means entity 2 (Kim Jong Un) is
  the Attacker argument of event 0 ("detonate": Conflict:Attack), and the local
  score is 0.8.
