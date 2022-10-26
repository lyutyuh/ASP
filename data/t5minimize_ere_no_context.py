from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import copy
import collections
import logging

from collections import defaultdict

from transformers import T5Tokenizer

# Usage:
# python t5minimize_ere_no_context.py true ./conll04/ ./conll04/

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

MENTION_START = '<m>'
MENTION_END   = '</m>'


def get_target_sentences(
    mentions, sentence, inv_subtoken_map, subtoken_map,
    entity_labels, m_special_start, m_special_end
):
    m_end_instantiations = [m_special_end.format(t) for t in entity_labels.keys()]

    if len(mentions) > 0:
        m_types = [x['type'] for x in mentions]
        m_startings = [x['start'] for x in mentions]
        m_endings = [x['end'] for x in mentions]
    else:
        m_types, m_startings, m_endings = [], [], []
    sorted_pos = sorted(
        [(inv_subtoken_map[x][0], m_special_end.format(t), entity_labels[t], ind) for ind,(x,t) in enumerate(zip(m_endings,m_types))] + \
        [(inv_subtoken_map[x][0], m_special_start, t, ind) for ind,(x,t) in enumerate(zip(m_startings,m_types))],
        reverse=True
    )
    
    target_sentence = copy.deepcopy(sentence)
    ent_indices = [-1 for i in range(len(sentence))]
    ent_type_sequence = [-1 for i in range(len(sentence))]
    target_subtoken_map = copy.deepcopy(subtoken_map)

    end_to_index_in_target = {}

    for x in sorted_pos:
        target_sentence.insert(x[0], x[1])
        ent_indices.insert(x[0], x[3])
        if x[1] in m_end_instantiations:
            target_subtoken_map.insert(x[0], subtoken_map[x[0]-1])
        elif x[1] == m_special_start:
            target_subtoken_map.insert(x[0], subtoken_map[x[0]+1])
        
        for k in end_to_index_in_target:
            end_to_index_in_target[k] += 1
        end_to_index_in_target[x[0]] = x[0]
        
        if x[1] in m_end_instantiations:
            ent_type_sequence.insert(x[0], x[2])
        else:
            ent_type_sequence.insert(x[0], -1)
        
    return (target_sentence, ent_indices, 
            ent_type_sequence, end_to_index_in_target,
            target_subtoken_map)

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
    processed_instances = []
    max_targ_len = 0
    max_inp_len = 0

    rel_id_int = {}
    for rel_type in relation_labels.keys():
        num_rel_now = sum(len(x) for x in rel_id_int.values())
        rel_id_int[rel_type] = (num_rel_now, ) if relation_labels[rel_type][1] else (
            num_rel_now, num_rel_now + 1)
        

    with open(input_path, "r") as input_file:
        instances = json.loads(input_file.read())
        
        for ins_id, instance in enumerate(instances):
            processed, subtoken_map = [], []
            
            inv_subtoken_map = {}
            tokens = instance['tokens']
            entities = instance['entities']
            relations = instance['relations']            
            
            first_token_in_doc = True
            after_hyphen = False
            word_idx = -1
            for word in tokens:
                if first_token_in_doc:
                    # insert prefix
                    prefix_text = tokenizer.tokenize("relation extraction:")
                    for sidx, subtoken in enumerate(prefix_text):
                        processed.append(subtoken)
                        subtoken_map.append(word_idx)

                word_idx += 1
                word = normalize_word(word, "english")
                    
                if word == "(" or word == "[":
                    subtokens = tokenizer.tokenize(word)
                elif word in [")", "]", "\'"]:
                    subtokens = tokenizer.tokenize(word)[1:] # skipping '_'
                elif is_punctuation(word):
                    subtokens = tokenizer.tokenize(word)[1:] # skipping '_'
                else:
                    subtokens = tokenizer.tokenize(word)

                inv_subtoken_map[word_idx] = (len(processed), len(processed)+len(subtokens))

                for sidx, subtoken in enumerate(subtokens):
                    processed.append(subtoken)
                    subtoken_map.append(word_idx)
                
                first_token_in_doc = False
                max_inp_len = max(max_inp_len, len(processed))
                pass
            
            inv_subtoken_map[word_idx+1] = (len(processed), len(processed)+1)
            processed.append("</s>")
            subtoken_map.append(word_idx+1)
            
            target_sentence, ent_indices, ent_type_sequence, end_to_index_in_target, target_subtoken_map = get_target_sentences(
                entities, processed, inv_subtoken_map, subtoken_map,
                entity_labels, MENTION_START, MENTION_END
            )

            if "sentence_idx" in instance:
                def clamp(x, l, u):
                    return max(min(x, u), l)
                sentence_idx = [instance['sentence_idx'][clamp(x, 0, len(instance['sentence_idx'])-1)] for x in subtoken_map]
                target_sentence_idx = [instance['sentence_idx'][clamp(x, 0, len(instance['sentence_idx'])-1)] for x in target_subtoken_map]
            
            max_targ_len = max(max_targ_len, len(target_sentence))
            rel_type_sequence, rel_indices = [[] for i in range(len(target_sentence))], [[] for i in range(len(target_sentence))]
            
            for rel in relations:
                head, tail = rel['head'], rel['tail']
                rel_type = rel['type']
                rel_symm = relation_labels[rel_type][1]

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
                "subtoken_map": subtoken_map,
                "target_sentence": target_sentence,
                "ent_type_sequence": ent_type_sequence,
                "rel_type_sequence": rel_type_sequence,
                "ent_indices": ent_indices,
                'rel_indices': rel_indices
            }
            if "sentence_idx" in instance:
                res['sentence_idx'] = sentence_idx
                res['target_sentence_idx'] = target_sentence_idx
                assert len(res['sentence_idx']) == len(res['sentence'])
                assert len(res['target_sentence_idx']) == len(res['target_sentence'])
            processed_instances.append(res)
        
    with open(output_path, "w") as output_file:
        output_file.write(json.dumps(processed_instances))

    print("Maximum input sequence length: {}".format(max_inp_len))
    print("Maximum target sequence length: {}".format(max_targ_len))
    print("Wrote {} sentences to {}".format(len(processed_instances), output_path))


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
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]

    if "conll04" in input_dir:
        typefile = input_dir + "conll04_types.json"
    elif "ace05" in input_dir:
        typefile = input_dir + "ace05_types.json"

    with open(typefile) as fin:
        labels = json.loads(fin.read())
    
    entity_labels, relation_labels = {}, {}
    
    for k in labels['entities'].keys():
        entity_labels[k] = len(entity_labels)
    for k in labels['relations'].keys():
        relation_labels[k] = (len(relation_labels),
                              labels["relations"][k]["symmetric"])

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    stats = defaultdict(int)
    minimize_language(
        entity_labels, relation_labels, stats, 
        input_dir, output_dir
    )
    print("stats:", stats)
