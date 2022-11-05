from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import copy
import tempfile
import subprocess
import collections
from typing import Optional, Tuple, Any, Dict, Iterable, List

from collections import defaultdict
from unittest import result

from transformers import T5Tokenizer
import truecase

# Usage:
# python t5minimize_ere.py ./ace05/ ./ace05/

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)

MENTION_START = '<m>'
MENTION_END   = '</m>'

def get_target_sentences(
    mentions: List[Tuple[int, int]], # list of (start, end) indices in the original sentence
    sentence: List[str], # tokenized sentence tokens
    inv_subtoken_map: List[Tuple[int, int]], # mapping a token to (start, end) of tokenized subtoken
    subtoken_map: List[int], # mapping a subtoken to original token
    entity_labels: Dict[str, int], # mapping an entity label to int label
    m_special_start: str, # opening bracket for mentions
    m_special_end: str, # closing bracket for mentions
):
    if len(mentions) > 0:
        m_types = [x['type'] for x in mentions]
        m_startings = [x['start'] for x in mentions]
        m_endings = [x['end'] for x in mentions]
    else:
        m_types, m_startings, m_endings = [], [], []


    sorted_pos = sorted(
        [(inv_subtoken_map[x][0], m_special_end, entity_labels[t], ind) for ind,(x,t) in enumerate(zip(m_endings,m_types))] + \
        [(inv_subtoken_map[x][0], m_special_start, t, ind) for ind,(x,t) in enumerate(zip(m_startings,m_types))],
        reverse=True
    )
    # when inserting positions are the same, the closing bracket comes first
    # which means that the closing bracket is inserted first
    # and the opening bracket is inserted later
    
    target_sentence = copy.deepcopy(sentence)
    ent_indices = [-1 for i in range(len(sentence))]
    ent_type_sequence = [-1 for i in range(len(sentence))]
    target_subtoken_map = copy.deepcopy(subtoken_map)

    end_to_index_in_target = {}

    for x in sorted_pos:
        target_sentence.insert(x[0], x[1]) # insert bracket
        ent_indices.insert(x[0], x[3]) # insert index of entity. 
        # opening and closing brackets of the same entity have the same index

        if x[1] == m_special_end: # insert entity type
            ent_type_sequence.insert(x[0], x[2])
        else:
            ent_type_sequence.insert(x[0], -1)

        for k in end_to_index_in_target: # map index of token in src to index in target
            # plus 1 for every special token inserted
            end_to_index_in_target[k] += 1
        end_to_index_in_target[x[0]] = x[0]

        if x[1] == m_special_end:
            target_subtoken_map.insert(x[0], subtoken_map[x[0]-1])
        elif x[1] == m_special_start:
            target_subtoken_map.insert(x[0], subtoken_map[x[0]+1])

    return (
        target_sentence,
        ent_indices,
        ent_type_sequence,
        end_to_index_in_target,
        target_subtoken_map
    )


def is_punctuation(c):
    if (
        c in {".", ",", "?", "!", ";", ":", "'s", "'m", "'ve", "n't", "'ll"}
    ):
        return True
    return False


def is_special(c):
    if (
        c in {"<pad>", "</s>", "<unk>"}
    ):
        return True
    return False


def get_doc_subtokens(doc):
    processed_doc = []
    first_token_in_doc = True
    for word in doc:
        if first_token_in_doc:
            # insert prefix
            prefix_text = tokenizer.tokenize("relation extraction:")
            for sidx, subtoken in enumerate(prefix_text):
                processed_doc.append(subtoken)

        subtokens = get_subtokens(word)
        for sidx, subtoken in enumerate(subtokens):
            processed_doc.append(subtoken)
        first_token_in_doc = False
    processed_doc.append("</s>")

    return processed_doc


def minimize_partition(
    name, entity_labels, relation_labels, stats,
    tokenizer, input_dir, output_dir
):
    if "conll04" in input_dir:
        input_path = f"{input_dir}/conll04_{name}.json"
    elif "ace05" in input_dir:
        input_path = f"{input_dir}/ace05_{name}.json"
    output_path = f"{output_dir}/{name}.t5-small.jsonlines"

    print("Minimizing {}".format(input_path))
    processed_dataset = []
    max_target_len, max_input_len = 0, 0

    rel_id_int = {}
    for rel_type in relation_labels.keys():
        num_rel_now = sum(len(x) for x in rel_id_int.values())
        rel_id_int[rel_type] = (num_rel_now, ) if relation_labels[rel_type][1] else (
            num_rel_now, num_rel_now + 1)

    with open(input_path, "r") as input_file:
        instances = json.load(input_file)

        tc = truecase.get_truecaser()
        for ins_id, instance in enumerate(instances):
            processed, subtoken_map = [], []
            inv_subtoken_map = {}

            tokens = instance['tokens']
            entities = instance['entities']
            relations = instance['relations']

            if 'extended' in instance:
                extended = instance['extended']
            else:
                extended = copy.deepcopy(instance['tokens'])

            if "".join(tokens).lower() == "".join(tokens) and "ace05" in input_dir: # no capitalization
                if '<extra_id_23>' in tokens:
                    # Marker: where should we copy from?
                    tok_st, tok_ed = tokens.index("<extra_id_22>"), tokens.index("<extra_id_23>")
                    # restore capitalization
                    tokens[tok_st+1:tok_ed] = tc.get_true_case_from_tokens(tokens[tok_st+1:tok_ed], out_of_vocabulary_token_option="as-is")
                else:
                    tokens = tc.get_true_case_from_tokens(tokens, out_of_vocabulary_token_option="as-is")

                if '<extra_id_23>' in extended:
                    ext_st, ext_ed = extended.index("<extra_id_22>"), extended.index("<extra_id_23>")
                    extended[:ext_st] = tc.get_true_case_from_tokens(extended[:ext_st], out_of_vocabulary_token_option="as-is")
                    extended[ext_st+1:ext_ed] = tc.get_true_case_from_tokens(extended[ext_st+1:ext_ed], out_of_vocabulary_token_option="as-is")
                    extended[ext_ed+1:] = tc.get_true_case_from_tokens(extended[ext_ed+1:], out_of_vocabulary_token_option="as-is")
                else:
                    extended = tc.get_true_case_from_tokens(extended, out_of_vocabulary_token_option="as-is")

            word_idx = -1

            for word in tokens:
                # no prefix inserted here
                word_idx += 1

                subtokens = get_subtokens(word)
                inv_subtoken_map[word_idx] = (len(processed), len(processed)+len(subtokens))

                for sidx, subtoken in enumerate(subtokens):
                    processed.append(subtoken)
                    subtoken_map.append(word_idx)

            inv_subtoken_map[word_idx+1] = (len(processed), len(processed)+1)
            processed.append("</s>")
            subtoken_map.append(word_idx+1)
            
            target_sentence, ent_indices, ent_type_sequence, end_to_index_in_target, target_subtoken_map = get_target_sentences(
                entities, processed, 
                inv_subtoken_map, subtoken_map,
                entity_labels, MENTION_START, MENTION_END
            )
            
            max_target_len = max(max_target_len, len(target_sentence))
            rel_type_sequence, rel_indices = [[] for i in range(len(target_sentence))], [[] for i in range(len(target_sentence))]
            # relation type sequence and antecedent indices
            
            for rel in relations:
                head, tail = rel['head'], rel['tail'] # indices of head and tail entities
                rel_type = rel['type']
                rel_symm = relation_labels[rel_type][1]

                # right boundary of head and tail entity
                head_rb = end_to_index_in_target[inv_subtoken_map[entities[head]['end']][0]]
                tail_rb = end_to_index_in_target[inv_subtoken_map[entities[tail]['end']][0]]
                
                if head_rb < tail_rb: # head before tail
                    rel_id = rel_id_int[rel_type][0]
                    for i in range(len(rel_indices)):
                        if i == tail_rb:
                            rel_indices[i].append(head_rb)
                            rel_type_sequence[i].append(rel_id)
                            stats[rel_id] += 1
                        else:
                            rel_indices[i].append(-1)
                            rel_type_sequence[i].append(-1)
                else:  # tail before head
                    rel_id = rel_id_int[rel_type][0] if rel_symm else rel_id_int[rel_type][1]
                    for i in range(len(rel_indices)):
                        if i == head_rb:
                            rel_indices[i].append(tail_rb)
                            rel_type_sequence[i].append(rel_id)
                            stats[rel_id] += 1
                        else:
                            rel_indices[i].append(-1)
                            rel_type_sequence[i].append(-1)
                            
            res = {
                "doc_id": name+"_"+str(ins_id),
                "sentence": processed, 
                # sentence is only for copying, could be different from 
                # input_sentence which is for encoding, where we might insert prefix, context, etc.
                "input_sentence": get_doc_subtokens(doc=extended),
                "subtoken_map": subtoken_map,
                "target_sentence": target_sentence,
                "ent_type_sequence": ent_type_sequence,
                "rel_type_sequence": rel_type_sequence,
                "ent_indices": ent_indices,
                'rel_indices': rel_indices
            }
            max_input_len = max(max_input_len, len(res['input_sentence']))

            processed_dataset.append(res)

    with open(output_path, "w") as output_file:
        json.dump(processed_dataset, output_file)


    print("Maximum input sequence length: {}".format(max_input_len))
    print("Maximum target sequence length: {}".format(max_target_len))
    print("Wrote {} sentences to {}".format(len(processed_dataset), output_path))


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``": # <unk> otherwise
        return "\""
    elif word == "`": # <unk> otherwise
        return "\'"
    else:
        return word


def get_subtokens(word):
    word = normalize_word(word, "english")
    if word == "(" or word == "[":
        subtokens = tokenizer.tokenize(word)
    elif word in [")", "]", "\'"]:
        subtokens = tokenizer.tokenize(word)[1:] # skipping '_'
    elif is_punctuation(word):
        subtokens = tokenizer.tokenize(word)[1:] # skipping '_'
    else:
        subtokens = tokenizer.tokenize(word)
    return subtokens


def minimize_language(
    entity_labels, relation_labels, stats,  
    input_dir, output_dir
):
    # including typed markers
    tokenizer.add_tokens(MENTION_START)
    tokenizer.add_tokens(MENTION_END)

    if "conll04" in input_dir:
        for name in ["dev", "test", "train_dev", "train"]:
            minimize_partition(
                name, entity_labels, relation_labels, 
                stats, tokenizer, input_dir, output_dir
            )
    elif "ace05" in input_dir:
        for name in ["dev", "test", "train"]:
            minimize_partition(
                name, entity_labels, relation_labels, 
                stats, tokenizer, input_dir, output_dir
            )
    return
    
if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if "conll04" in input_dir:
        typefile = f"{input_dir}/conll04_types.json"
    elif "ace05" in input_dir:
        typefile = f"{input_dir}/ace05_types.json"

    with open(typefile) as input_file:
        labels = json.load(input_file)
    
    entity_labels, relation_labels = {}, {}
    
    for k in labels['entities'].keys():
        entity_labels[k] = len(entity_labels)
    for k in labels['relations'].keys():
        relation_labels[k] = (len(relation_labels), labels["relations"][k]["symmetric"])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    stats = defaultdict(int)
    minimize_language(
        entity_labels, relation_labels, stats, 
        input_dir, output_dir
    )
    print("stats:", stats)
