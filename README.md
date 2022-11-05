# Autoregressive Structured Prediction with Language Models
This repository contains PyTorch implementation and pre-trained models for ``ASP``, described in [Autoregressive Structured Prediction with Language Models](https://arxiv.org/pdf/2210.14698.pdf).

<p align="right" width="100%"> Links: <a href="https://github.com/eth-nlped">ETH-NLPED lab</a> , <a href="https://github.com/rycolab">Rycolab</a>  </p>

![](./figs/illustration.gif)

## Contents
* [Setup](#Setup)
  * [Installation](#clone-this-repo)
  * [Virtual environment](#Prepare-the-environment)
  * [Download datasets](#Download-and-preprocess-the-datasets)
* [Tasks](#Tasks)
* [Pre-trained Models](#Pre-trained-Models)
* [Citation](#Citation)


## Setup

### 1. Clone this repo:
```bash
git clone https://github.com/lyutyuh/ASP.git
cd ASP
export ASP=$PWD # setting environment variable
```
### 2. Prepare the environment

#### 2.1 Create virtual environment with:
<details>
    <summary> <code> pip </code> </summary>

```bash
python -m venv <path_to_venv>/asp    # create a new environment (asp)
source <path_to_venv>/asp/bin/activate
pip install -r requirements.txt
```

</details> 
or

<details>
    <summary> <code>conda</code> </summary>

```bash
conda env create -f environment.yml    # create a new environment (asp)
```

</details>


#### 2.2 (Optional) To use ```FusedAdam``` to speed up training (by ~30%)

<details>
    <summary> Install <code>apex</code> from source </summary>

  ```bash
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

</details>



## Download and preprocess the datasets


<details>
    <summary> <code> named entity recognition </code> </summary>
    
### CoNLL-03
```bash
  wget https://polybox.ethz.ch/index.php/s/bFf8vJBonIT7sr8/download -O ./data/conll03_ner.zip
  unzip ./data/conll03_ner.zip -d ./data
  rm ./data/conll03_ner.zip
  python ./data/conll03_ner/conll03_to_json.py
  python ./data/t5minimize_ner.py ./data/conll03_ner ./data/conll03_ner
```

### OntoNotes V5
Coming soon!
</details> 


<details>
    <summary> <code> end-to-end relation extraction </code> </summary>

### CoNLL-04

  ```bash
    wget https://polybox.ethz.ch/index.php/s/Lk44AwhOeDSeZTh/download -O ./data/conll04_ere.zip
    unzip ./data/conll04_ere.zip -d ./data
    rm ./data/conll04_ere.zip
    python ./data/t5minimize_ere.py ./data/conll04_ere/ ./data/conll04_ere
  ```

### ACE-05

ACE-05 is not a publically available dataset. Please follow https://github.com/luanyi/DyGIE/tree/master/preprocessing to obtain
the dataset json files ```{train,dev,test}.json``` and copy them to ```./data/ace05_ere/```.

Then:

  ```bash
    python ./data/ace05_ere/ace05_to_json.py
    python ./data/t5minimize_ere.py ./data/ace05_ere ./data/ace05_ere
  ```

</details> 


<details>
    <summary> <code> coreference resolution </code> </summary>

### CoNLL-12 (OntoNotes)

OntoNotes is not a publically available dataset. Please follow http://conll.cemantix.org/2012/data.html and https://catalog.ldc.upenn.edu/LDC2013T19 to obtain
the files ```{train,dev,test}.english.v4_gold_conll``` and copy them to ```./data/ontonotes_coref/```.

Then:

  ```bash
  python ./data/t5minimize_coref.py ./data/ontonotes_coref/ ./data/ontonotes_coref/
  ```

</details> 


## Tasks

For task in ```{ner,ere,coref}```:
```bash
  python run_{task}.py <config_name> 0 
```
Please find the ```<config_name>``` in each ```{ner,ere,coref}.conf``` file under [configs](configs)



## Running on New Datasets
### 1. prepare the data
* For `named entity recognition` and `relation extraction`,
convert the new dataset to `<newdataset>_{train,dev,test}.json` in the following format:
```json
[{
    "tokens": ["John", "Wilkes", "Booth", ",", "who", "assassinated", "President", "Lincoln", ",", "was", "an", "actor", "."], 
    "entities": [{"type": "Peop", "start": 0, "end": 3}, {"type": "Peop", "start": 6, "end": 8}], 
    "relations": [{"type": "Kill", "head": 0, "tail": 1}] // Not necessary for NER
}, ...]
```
and `<newdataset>_types.json`:

```json
{
    "entities": {
        "Loc": {"short": "Loc", "verbose": "Location"}, 
        "Org": {"short": "Org", "verbose": "Organization"}, 
        "Peop": {"short": "Peop", "verbose":"People"}, 
        "Other": {"short": "Other", "verbose": "Other"}
    }, 
    "relations": { // Not necessary for NER
        "Work_For": {"short": "Work", "verbose": "Work for", "symmetric": false}, 
        "Kill": {"short": "Kill", "verbose": "Kill", "symmetric": false}, 
        "OrgBased_In": {"short": "OrgBI", "verbose": "Organization based in", "symmetric": false}, 
        "Live_In": {"short": "Live", "verbose": "Live in", "symmetric": false}, 
        "Located_In": {"short": "LocIn", "verbose": "Located in", "symmetric": false}
    }
}
```


and run 
```bash
  python ./data/t5minimize_ere.py ./data/<newdataset>/ ./data/<newdataset>/
```


* For coreference resolution, convert the new dataset to CoNLL-12 format.
Then
```bash
python ./data/t5minimize_coref.py ./data/<newdataset>/ ./data/<newdataset>/
```

### 2. Prepare the configuration
Add a new entry in the corresponding `.conf` file under [configs](configs) with the directory to the new dataset `data_dir = ${ASP}/data/<newdataset>/`


## Pre-trained models



## Citation
```bibtex
@inproceedings{liu-etal-2022-autoregressive,
    title={Autoregressive Structured Prediction with Language Models},
    author={Tianyu Liu and Yuchen Jiang and Nicholas Monath and Ryan Cotterell and Mrinmaya Sachan},
    year={2022},
    url={https://arxiv.org/abs/2210.14698},
    eprint={2210.14698},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```