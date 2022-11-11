import json
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)


def conll03_to_json():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"converting conll03 to json in {dir_path}")

    conll03_datasets, conll03_types = {}, {}

    for name in ["dev", "test", "train"]:
        logger.info(f"processing {dir_path}/{name}.txt")
        data = open(f"{dir_path}/{name}.txt").readlines()
        
        dataset = []
        idx, start, current_type, doc = -1, None, None, None
        for line in data:
            line = line.strip()

            if line == "-DOCSTART- -X- -X- O": # new doc
                if doc is not None:
                    # when extended is not the same as tokens
                    # mark where to copy from with <extra_id_22> and <extra_id_23>
                    # E.g.
                    # Extract entities such as apple, orange, lemon <extra_id_22> Give me a mango . <extra_id_23>
                    # See ace05_to_json.py for example of extending the input
                    doc["extended"] = doc["tokens"]
                    dataset.append(doc)
                doc = {
                    "tokens": [], # list of tokens for the model to copy from
                    "extended": [], # list of input tokens. Prompts, instructions, etc. go here
                    "entities": [] # list of dict:{"type": type, "start": start, "end": end}, format: [start, end)
                }
                idx, start = -1, None
                continue
            elif line == "":
                if len(doc["tokens"]) > 800 and name == "train": # clip
                    if doc is not None:
                        doc["extended"] = doc["tokens"]
                        dataset.append(doc)
                    doc = {
                        "tokens": [],
                        "extended": [],
                        "entities": []
                    }
                    idx, start = -1, None
                    continue
                # new sentence
                pass
            else:
                idx += 1
                items = line.split()
                assert len(items) == 4, line

                token, _, _, bio_tag = items
                doc["tokens"].append(items[0])

                if bio_tag[0] == 'I':
                    pass
                elif bio_tag[0] == 'O':
                    if start is not None:
                        doc['entities'].append(
                            {
                                "type": current_type,
                                "start": start,
                                "end": idx
                            }
                        )
                    start = None
                elif bio_tag[0] == 'B':
                    if start is not None:
                        doc['entities'].append(
                            {
                                "type": current_type,
                                "start": start,
                                "end": idx
                            }
                        )
                    start = idx
                    current_type = bio_tag[2:]
                    conll03_types[current_type] = {
                        "short": current_type
                    }
        dataset.append(doc)
        conll03_datasets[name] = dataset
    for name in conll03_datasets:
        logger.info(f"maximum input length: {max([len(x['extended']) for x in conll03_datasets[name]])}")
        logger.info(f"saving {len(conll03_datasets[name])} documents to {dir_path}/conll03_{name}.json")
        with open(f"{dir_path}/conll03_{name}.json", 'w') as fout:
            json.dump(conll03_datasets[name], fout)
    with open(f"{dir_path}/conll03_types.json", 'w') as fout:
        logger.info(f"saving types to {dir_path}/conll03_types.json")
        json.dump({"entities": conll03_types}, fout)
    return

if __name__ == "__main__":
    conll03_to_json()