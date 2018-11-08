import json
import re
import argparse
from typing import Callable, Dict, List

import spacy
from allennlp.common.util import JsonDict
from allennlp.pretrained import (srl_with_elmo_luheng_2018,
                                 open_information_extraction_stanovsky_2018)

SPACY_NLP = spacy.load('en')

def make_table_files(data_file: str,
                     output_path: str,
                     tagger: Callable[str, JsonDict]):
    data = json.load(open(data_file))
    for paragraph_id, paragraph_data in data.items():
        passage = paragraph_data['passage']
        processed_doc = SPACY_NLP(passage)
        sentences: List[str] = []
        table_info: List[Dict[str, str]] = []
        column_names = ['sentence_id', 'verb']
        for sentence_id, sentence in enumerate(processed_doc.sents):
            sentences.append(sentence)
            pas_info = tagger(sentence)
            for verb_info in pas_info['verbs']:
                verb = verb_info['verb']
                row_info = {'verb': verb}
                pas_parts = re.findall(r'\[[^]]*\]', verb_info['description'])
                for pas_part in pas_parts:
                    pas_part = pas_part[1:-1]  # remove opening and closing brackets
                    pas_key, pas_value = pas_part.split(": ")
                    if pas_key == "V":
                        continue
                    if pas_key not in column_names:
                        column_names.append(pas_key)
                    row_info[pas_key] = pas_value
                row_info['sentence_id'] = sentence_id
                table_info.append(row_info)

        with open(f"{output_path}/{paragraph_id}.table", "w") as output_file:
            for row_info in table_info:
                print("\t".join([row_info[column_name] if column_name in row_info else "NULL"
                                 for column_name in column_names]), file=output_file)

        with open(f"{output_path}/{paragraph_id}.sentences", "w") as output_file:
            for sentence_id, sentence in enumerate(sentences):
                print(f"{sentence_id}\t{sentence}", file=output_file)


def main(args):
    if args.tagger == "srl":
        model = srl_with_elmo_luheng_2018()
        tagger_function = model.predict
    elif args.tagger == "oie":
        model = open_information_extraction_stanovsky_2018()
        tagger_function = lambda sentence: model.predict_json({"sentence": sentence})
    else:
        raise RuntimeError(f"Unknown tagger type: {args.tagger}")
    make_table_files(args.data_file, args.output_path, tagger_function)


if __name__ == '__main__':
    # pylint: disable=invalid-name
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_file", type=str, help="Path to input data file in JSON format")
    argparser.add_argument("output_path", type=str, help="Path to the output directory")
    argparser.add_argument("tagger", type=str, help="Type of tagger to use")
    arguments = argparser.parse_args()
    main(arguments)
