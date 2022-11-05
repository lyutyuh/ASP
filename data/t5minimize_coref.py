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
from typing import Optional, Tuple, Any, Dict, Iterable, List

import util
import conll
from transformers import T5Tokenizer

# Usage:
# python t5minimize_coref.py ontonotes_coref/ ontonotes_coref/

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)

SPEAKER_START = '<speaker>'
SPEAKER_END   = '</speaker>'
MENTION_START = '<m>'
MENTION_END   = '</m>'

prefix_subtokens = tokenizer.tokenize("coreference resolution:")
prefix_len = len(prefix_subtokens)

tokenizer.add_tokens(SPEAKER_START)
tokenizer.add_tokens(SPEAKER_END)
tokenizer.add_tokens(MENTION_START)
tokenizer.add_tokens(MENTION_END)


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.segment_info = []


    def finalize(self):
        # populate clusters
        first_subtoken_index = -1
        mention_to_seg_id = {}
        for seg_idx, segment in enumerate(self.segment_info):
            # keeping all segments
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else '-'
                if coref != "-":
                    last_subtoken_index = first_subtoken_index + \
                        tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
                                mention_to_seg_id[first_subtoken_index] = seg_idx
                                mention_to_seg_id[last_subtoken_index+1] = seg_idx
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(
                                    first_subtoken_index)
                        else:
                            cluster_id = int(part[:-1])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append((start, last_subtoken_index))
                            mention_to_seg_id[start] = seg_idx
                            mention_to_seg_id[last_subtoken_index+1] = seg_idx

        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                logger.info("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))

        # merged_clusters: list of clusters
        merged_clusters = [list(c) for c in merged_clusters]
        cluster_indices = {
            x: i for i in range(len(merged_clusters)) for x in merged_clusters[i]}

        docs = []
        num_words = len(util.flatten(self.segments))
        num_segments = len(self.segments)

        subtoken_map = self.segment_subtoken_map
        assert num_words == len(util.flatten(self.segment_subtoken_map))

        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        assert num_words == len(sentence_map), (num_words, len(sentence_map))

        all_mentions = util.flatten(merged_clusters)
        assert len(all_mentions) == len(set(all_mentions))
        sentences = self.segments

        # inserting <m> and </m> into target sequences for all mentions
        target_sentences = m_star_target_sequences(
            all_mentions, self.segments,
            MENTION_START, MENTION_END,
            mention_to_seg_id
        )

        # inserting mention indices to <\m>
        mention_indices = m_star_insert_info(
            all_mentions, self.segments,
            [ix for ix, x in enumerate(all_mentions)],
            mention_to_seg_id
        )
        mention_indices = post_processing_mention_indices(mention_indices)
        # inserting cluster indices to <\m>
        cluster_categories = m_star_insert_info(
            all_mentions, self.segments,
            [cluster_indices[x] for x in all_mentions],
            mention_to_seg_id
        )
        for i in range(len(cluster_categories)):
            cluster_categories[i] = [
                x if type(x) != list else -1 for x in cluster_categories[i]]
            assert len(cluster_categories[i]) == len(target_sentences[i]) == len(mention_indices[i])

        clusters = [[] for _ in range(len(self.segments))]
        for x in all_mentions:
            clusters[mention_to_seg_id[x[0]]].append(x)

        for i in range(num_segments):
            docs.append({
                "doc_key": f'{self.doc_key}_{i}',
                "sentence": sentences[i],
                "target_sentence": target_sentences[i],
                "mention_indice": mention_indices[i],
                "cluster_category": cluster_categories[i],
                'sentence_map': sentence_map[i],
                "subtoken_map": subtoken_map[i]
            })

        return docs


def post_processing_mention_indices(
    mention_indices
):
    """
    Post-processing mention indices.
    E.g.
        [ q  <m>  a  b <\m>  c <\m> ]
        [-1  -1  -1 -1   1  -1   1  ]
    """
    tmp_mention_indices = []
    for seg_i in range(len(mention_indices)):
        tmp_mention_indices_seg_i = []
        for j in range(len(mention_indices[seg_i])):
            # reading from left to right
            if type(mention_indices[seg_i][j]) != list:
                # j is either 1. word or 2. closing bracket
                tmp_mention_indices_seg_i.append(mention_indices[seg_i][j])
                # putting the index of pairing opening bracket or -1
                continue
            else:
                # j is opening bracket [*
                tmp_mention_indices_seg_i.append(-1)
                for k in range(j+1, len(mention_indices[seg_i])):
                    if mention_indices[seg_i][k] in mention_indices[seg_i][j]:
                        # the closing bracket k pairs with opening bracket j
                        mention_indices[seg_i][k] = j
        tmp_mention_indices.append(tmp_mention_indices_seg_i)
    return tmp_mention_indices


def m_star_insert_info(
    mentions: List[Tuple[int, int]],
    segments: List[List[str]], 
    m_infos: List[int], 
    mention_to_seg_id: Dict[int, int]
):
    """
        Get a sequence of information of the same length with the target sequence.
        mentions: list of mentions, e.g. [(0, 0), (2, 3), (4, 4)] format: [start, end] (inclusive)
        segments: list of segments, e.g. [['I', 'have', 'a', 'cat'], ['I', 'have', 'a', 'dog']]
        m_infos: list of information to be inserted with each mention, 
                 e.g. cluster indices [0, 1, 2]
        mention_to_seg_id: dict, mapping mention to its segment id
    """
    m_startings, m_endings = zip(*mentions) if len(mentions) > 0 else ([], [])
    # order preserving
    sorted_pos = sorted(
        [(x+1, -1, y) for x, y in zip(m_endings, m_infos)] +
        [(x,  1, [y]) for x, y in zip(m_startings, m_infos)],
        reverse=True # insert from right to left, so that the calculated positions are not changed
    ) 
    # when inserting positions are the same, the closing bracket comes first
    # which means that the closing bracket is inserted first
    # and the opening bracket is inserted later

    target_sequences = [
        [-1 for x in range(len(segments[i]))] for i in range(len(segments))]
    # offset of each segment
    offsets = list(accumu([len(x) for x in segments]))

    prev_loc, prev_token = -1, None
    for x in sorted_pos:
        seg_idx = mention_to_seg_id[x[0]]
        offset = offsets[seg_idx]

        if x[0] == prev_loc and (x[1] == prev_token == 1): # 1 for starting
            # contracting left brackets
            target_sequences[seg_idx][x[0]-offset].extend(x[2])
        else:
            target_sequences[seg_idx].insert(x[0]-offset, x[2])
        prev_loc, prev_token = x[0], x[1]

    return target_sequences


def m_star_target_sequences(
    mentions: List[Tuple[int, int]],
    sequences: List[List[str]],
    m_special_start: str, 
    m_special_end: str,
    mention_to_seg_id: Dict[int, int]
):
    """
        Get a sequence of target sentences with <m> and <\m> inserted.
        mentions: list of mentions, e.g. [(0, 0), (2, 3), (4, 4)] format: [start, end] (inclusive)
        sequences: list of sequences, e.g. [['I', 'have', 'a', 'cat'], ['I', 'have', 'a', 'dog']]
        m_special_start: special token for starting bracket
        m_special_end: special token for ending bracket
        mention_to_seg_id: dict, mapping mention to its segment id
    """
    m_startings, m_endings = zip(*mentions) if len(mentions) > 0 else ([], [])
    sorted_pos = sorted(
        [(x+1, -1, m_special_end)   for x in m_endings] +
        [(x,    1, m_special_start) for x in m_startings],
        reverse=True # insert from right to left, so that the calculated positions are not changed
    )

    target_sequences = copy.deepcopy(sequences)
    # offset of each segment
    offsets = list(accumu([len(x) for x in sequences]))

    prev_loc, prev_token = -1, None
    for x in sorted_pos:
        seg_idx = mention_to_seg_id[x[0]]
        offset = offsets[seg_idx]

        if x[0] == prev_loc and (x[2] == prev_token == m_special_start):
            # contracting left brackets to [*
            pass # do nothing
        else:
            target_sequences[seg_idx].insert(x[0]-offset, x[2])
        prev_loc, prev_token = x[0], x[2]

    return target_sequences


def normalize_word(word, language):
    br_dict = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

    if language == "arabic":
        word = word[:word.find("#")]

    if word in br_dict:
        word = br_dict[word]
        return word
    elif word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``": # <unk> otherwise
        return "\""
    elif word == "`": # <unk> otherwise
        return "\'"
    else:
        return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(
    document_state, max_segment_len, constraints1, constraints2
):
    current = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 1 - prefix_len,
                  len(document_state.subtokens) - 1)

        while end >= current and not constraints1[end]:
            end -= 1

        if end < current:
            end = min(current + max_segment_len - 1 - 1 - prefix_len,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")

        document_state.segments.append(
            prefix_subtokens + document_state.subtokens[current:end+1] + ['</s>'])

        subtoken_map = document_state.subtoken_map[current:end+1]
        document_state.segment_subtoken_map.append(
            [subtoken_map[0]] * prefix_len + subtoken_map + [subtoken_map[-1]])
        document_state.segment_info.append(
            [None] * prefix_len + document_state.info[current:end+1] + [None])
        current = end + 1

    return


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) - 1 - prefix_len for s in segments])
    for segment in segments:
        sent_map.extend([current] * prefix_len)
        for i in range(len(segment) - 1 - prefix_len):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
        sent_map.append(current)
    return sent_map


def get_document(
    document_lines, tokenizer, language, segment_len
):
    document_state = DocumentState(document_lines[0])
    word_idx = -1

    current_speaker = None
    after_hyphen = False
    doc_lines = document_lines[1]

    for line in doc_lines:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            speaker_orthography = row[9].replace("_", " ").replace("#", " ").strip()
            if current_speaker is None or current_speaker != speaker_orthography:
                # insert speaker
                word_idx += 1
                current_speaker = speaker_orthography
                speaker_text = tokenizer.tokenize(current_speaker)
                document_state.tokens.append(current_speaker)

                for sidx, subtoken in enumerate([SPEAKER_START] + speaker_text + [SPEAKER_END]):
                    document_state.subtokens.append(subtoken)
                    info = None
                    document_state.info.append(info)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(word_idx)

            word_idx += 1
            word = normalize_word(row[3], language)

            if is_punctuation(word):
                subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
            elif after_hyphen:
                subtokens = tokenizer.tokenize("-"+word)  # skipping '_'
                if subtokens[1] == "-":
                    subtokens = subtokens[2:]
                else:
                    subtokens = subtokens[1:]
                after_hyphen = False
            else:
                subtokens = tokenizer.tokenize(word)

            if row[4] == "HYPH":
                after_hyphen = True

            document_state.tokens.append(word)
            document_state.token_end += [False] * (len(subtokens) - 1) + [True]

            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            document_state.sentence_end[-1] = True

    constraints1 = (
        document_state.sentence_end
        if language != "arabic"
        else document_state.token_end
    )
    split_into_segments(
        document_state, segment_len, constraints1, document_state.token_end
    )

    stats[f"max_seg_len"] = max(
        stats["max_seg_len"], max([len(s) for s in document_state.segments])
    )
    stats[f"max_num_seg"] = max(
        len(document_state.segments), stats[f"max_num_seg"]
    )
    document = document_state.finalize()
    return document


def is_punctuation(c):
    if (
        c in {".", ",", "?", "!", ";", 
        ":", "'s", "'m", "'ve", "n't", "'ll",
        ")", "]", "}", "-"}
    ):
        return True
    return False


def is_special(c):
    if (
        c in {"<pad>", "</s>", "<unk>"}
    ):
        return True
    return False


def accumu(lis):
    total = 0
    for x in lis:
        yield total
        total += x

def minimize_partition(
    name, language, extension, stats, tokenizer, seg_len, input_dir, output_dir
):
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    output_path = "{}/{}.t5-small.{}.{}.jsonlines".format(output_dir, name, language, seg_len)

    count = 0
    logger.info("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(
                    begin_document_match.group(1), begin_document_match.group(2)
                )
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)

    datasets, max_target_len = [], 0
    max_input_len = 0
    for document_lines in documents:
        max_input_len = max(max_input_len, len([x for x in document_lines[1] if len(x) > 2]))
        document = get_document(document_lines, tokenizer, language, seg_len)
        for doc in document:
            max_target_len = max([max_target_len] + [len(doc['cluster_category'])])
            datasets.append(doc)
            count += 1
    json.dump(datasets, open(output_path, "w"))
    logger.info(f"Maximum input sequence length: {max_input_len}, Maximum target sequence length: {max_target_len}")
    logger.info("Wrote {} documents to {}".format(count, output_path))


def minimize_language(language, stats, seg_len, input_dir, output_dir):
    minimize_partition("dev", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("train", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("test", language, "v4_gold_conll", stats,
                       tokenizer, seg_len, input_dir, output_dir)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for seg_len in [4096, 2048]:
        stats = collections.defaultdict(int)
        minimize_language("english", stats, seg_len, input_dir, output_dir)

        logger.info("Dataset stats:")
        for k, v in stats.items():
            logger.info("{} = {}".format(k, v))
