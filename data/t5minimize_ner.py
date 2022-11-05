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

from collections import defaultdict
from unittest import result

from transformers import T5Tokenizer

# Usage:
# python t5minimize_ner.py ./conll03_ner/ ./conll03_ner/

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)

MENTION_START = '<m>'
MENTION_END   = '</m>'

def get_target_sentences(
    mentions, sentence, 
    inv_subtoken_map, subtoken_map,
    entity_labels, m_special_start, m_special_end
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

    target_sentence = copy.deepcopy(sentence)
    ent_indices = [-1 for i in range(len(sentence))]
    ent_type_sequence = [-1 for i in range(len(sentence))]
    target_subtoken_map = copy.deepcopy(subtoken_map)

    end_to_index_in_target = {}

    for x in sorted_pos:
        target_sentence.insert(x[0], x[1]) # insert end or start
        ent_indices.insert(x[0], x[3]) # insert pairing bracket index for entity

        if x[1] == m_special_end: # insert entity type
            ent_type_sequence.insert(x[0], x[2])
        else:
            ent_type_sequence.insert(x[0], -1)

        for k in end_to_index_in_target: # map index in src to index in target
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
        c in {".", ",", "?", "!", ";", 
        ":", "'s", "'m", "'ve", "n't", "'ll",
        ")", "}", "]"}
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
    processed_doc, subtoken_map = [], []
    word_idx = -1
    first_token_in_doc = True
    for word in doc:
        word_idx += 1
        if first_token_in_doc:
            # insert prefix
            prefix_text = tokenizer.tokenize("named entity recognition:")
            for sidx, subtoken in enumerate(prefix_text):
                processed_doc.append(subtoken)
                subtoken_map.append(word_idx)

        subtokens = get_subtokens(word)
        for sidx, subtoken in enumerate(subtokens):
            processed_doc.append(subtoken)
            subtoken_map.append(word_idx)

        first_token_in_doc = False
    processed_doc.append("</s>")
    subtoken_map.append(word_idx+1)

    return processed_doc, subtoken_map


def minimize_partition(
    name, entity_labels, stats,
    tokenizer, input_dir, output_dir
):
    if "conll03" in input_dir:
        input_path = f"{input_dir}/conll03_{name}.json"

    output_path = f"{output_dir}/{name}.t5-small.jsonlines"

    print("Minimizing {}".format(input_path))
    processed_dataset = []
    max_target_len, max_input_len = 0, 0

    with open(input_path, "r") as input_file:
        instances = json.load(input_file)

        for ins_id, instance in enumerate(instances):
            processed, subtoken_map = [], []
            inv_subtoken_map = {}

            tokens = instance['tokens']
            entities = instance['entities']
            extended = instance['extended']

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
            if "sentence_idx" in instance:
                def clamp(x, l, u):
                    return max(min(x, u), l)
                sentence_idx = [instance['sentence_idx'][clamp(x, 0, len(instance['sentence_idx'])-1)] for x in subtoken_map]
                target_sentence_idx = [instance['sentence_idx'][clamp(x, 0, len(instance['sentence_idx'])-1)] for x in target_subtoken_map]
            
            max_target_len = max(max_target_len, len(target_sentence))
            input_sentence, input_subtoken_map = get_doc_subtokens(doc=extended)
            res = {
                "doc_id": name+"_"+str(ins_id),
                "sentence": processed, 
                # sentence is for copy mechanism, might be different from 
                # input_sentence which is for encoding only
                "input_sentence": input_sentence,
                "subtoken_map": subtoken_map,
                "target_sentence": target_sentence,
                "ent_type_sequence": ent_type_sequence,
                "ent_indices": ent_indices
            }
            max_input_len = max(max_input_len, len(res['input_sentence']))
            if "sentence_idx" in instance:
                res['sentence_idx'] = sentence_idx
                res['target_sentence_idx'] = target_sentence_idx
                assert len(res['sentence_idx']) == len(res['sentence'])
                assert len(res['target_sentence_idx']) == len(res['target_sentence'])
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
    elif word == "''" or word == "``":  # <unk> otherwise
        return "\""
    elif word == "`":  # <unk> otherwise
        return "\'"
    else:
        return word

def get_subtokens(word):
    word = normalize_word(word, "english")
    if word == "(" or word == "[":
        subtokens = tokenizer.tokenize(word)
    elif word in [")", "]", "\'"]:
        subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
    elif is_punctuation(word):
        subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
    else:
        subtokens = tokenizer.tokenize(word)
    return subtokens


def minimize_language(
    entity_labels, stats,
    input_dir, output_dir
):
    # including typed markers
    tokenizer.add_tokens(MENTION_START)
    tokenizer.add_tokens(MENTION_END)

    if "conll03" in input_dir:
        for name in ["dev", "test", "train"]:
            minimize_partition(
                name, entity_labels, stats, 
                tokenizer, input_dir, output_dir
            )
    return

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if "conll03" in input_dir:
        typefile = f"{input_dir}/conll03_types.json"

    with open(typefile) as input_file:
        labels = json.load(input_file)
    entity_labels = {}

    for k in labels['entities'].keys():
        entity_labels[k] = len(entity_labels)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    stats = defaultdict(int)
    minimize_language(
        entity_labels, stats, 
        input_dir, output_dir
    )
    print("stats:", stats)
