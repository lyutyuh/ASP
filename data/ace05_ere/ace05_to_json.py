import json
import copy
import os
import logging
from collections import defaultdict


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

dir_path = os.path.dirname(os.path.realpath(__file__))
logger.info(f"converting ace05 to json in {dir_path}")


def is_symmetric(rel_name):
    if rel_name == "PER-SOC":
        return True
    else:
        return False


def normalize_word(token):
    if token.lower() == '-lrb-':
        processed = '('
    elif token.lower() == '-rrb-':
        processed = ')'
    elif token.lower() == '-lsb-':
        processed = '['
    elif token.lower() == '-rsb-':
        processed = ']'
    elif token.lower() == "''":
        processed = '\"'
    elif token.lower() == '`':
        processed = "\'"
    elif token.lower() == '``':
        processed = '\"'
    else:
        processed = token
    return processed

ace05_dataset = {}

for dataset_type in ['train', 'dev', 'test']:
    with open(f'{dir_path}/{dataset_type}.json') as f:
        ace05_dataset[dataset_type] = []

        for line in f:
            doc = json.loads(line)
            ace05_dataset[dataset_type].append(doc)

ace05_dataset_asp = {}
types = {
    "entities": {},
    "relations": {}
}
stats = defaultdict(int)


for dataset_type in ['train', 'dev', 'test']:

    dataset = ace05_dataset[dataset_type]
    ace05_dataset_asp[dataset_type] = []
    dataset_fine = ace05_dataset_asp[dataset_type]

    for doc in dataset:
        sentences = doc['sentences']
        ner = doc['ner']
        relations = doc['relations']

        extended = [[normalize_word(token) for token in sentence] for sentence in sentences]

        bias = 0
        for i, (sentence, ne, relation) in enumerate(zip(sentences,ner,relations)):
            bias -= 1
            # marker: start of where to copy
            tokens = ['<extra_id_22>']

            for token in sentence:
                tokens.append(normalize_word(token))
            i_extended = copy.deepcopy(extended)
            # markers: end of where to copy
            i_extended.insert(i+1, ['<extra_id_23>'])
            i_extended.insert(i,   ['<extra_id_22>'])
            
            tokens.append('<extra_id_23>')
            
            window = 3
            upperbound_dict = {
                'train': 128,
                'dev': 256,
                'test': 256
            }
            while len(sum(i_extended[max(i-window, 0): min(i+3+window, len(i_extended)-1)], [])) > upperbound_dict[dataset_type] and\
                window >= 0:
                window -= 1

            item = {
                'tokens': tokens,
                'extended': sum(i_extended[
                    max(i-window, 0): min(i+3+window, len(i_extended)-1)
                ], []),
                'entities': [(i_begin-bias,i_end-bias+1,e) for i_begin, i_end, e in ne],
                'relations': [
                    (i_begin-bias,i_end-bias+1,j_begin-bias,j_end-bias+1, r) for \
                    i_begin, i_end, j_begin, j_end, r in relation
                ] 
            }

            span_to_id = {}
            for i, e in enumerate(item["entities"]):
                new_entity = {
                    "type": e[2],
                    "start": e[0],
                    "end": e[1]
                }
                span_to_id[(e[0], e[1])] = i
                item["entities"][i] = new_entity
                
                if not e[2] in types["entities"]:
                    types["entities"][e[2]] = {
                        "short": e[2]
                    }
                
            for i, r in enumerate(item["relations"]):
                stats[r[4]] += 1
                new_relation = {
                    "type": r[4],
                    "head": span_to_id[(r[0],r[1])],
                    "tail": span_to_id[(r[2],r[3])]
                }
                item["relations"][i] = new_relation
                
                if not r[4] in types["relations"]:
                    types["relations"][r[4]] = {
                        "short": r[4],
                        "symmetric": is_symmetric(r[4])
                    }
            dataset_fine.append(item)
            bias += len(tokens) - 1

    logger.info(f"{len(ace05_dataset_asp[dataset_type])} instances in {dataset_type}")
    with open(f'{dir_path}/ace05_{dataset_type}.json', 'w') as f:
        json.dump(dataset_fine, f)

with open(f'{dir_path}/ace05_types.json', 'w') as f:
    json.dump(types, f)
            
