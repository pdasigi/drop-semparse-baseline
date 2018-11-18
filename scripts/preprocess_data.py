import json
import re
import os
import sys
import argparse
import logging
from typing import Callable, Dict, List, Tuple

import spacy
from spacy.tokens import Span, Doc
from allennlp.common.util import JsonDict
from allennlp.pretrained import (srl_with_elmo_luheng_2018,
                                 open_information_extraction_stanovsky_2018)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from semparse.context.paragraph_question_context import MONTH_NUMBERS  # pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)

SPACY_NLP = spacy.load('en')


def get_table_info(processed_passage: Doc,
                   tagger: Callable[[str], JsonDict]) -> Tuple[List[Dict[str, str]], List[str], List[Span]]:
    processed_sentences: List[Span] = []
    table_info: List[Dict[str, str]] = []
    column_names = ['sentence_id', 'verb']
    for sentence_id, spacy_sentence in enumerate(processed_passage.sents):
        processed_sentences.append(spacy_sentence)
        pas_info = tagger(spacy_sentence.string.strip())
        for verb_info in pas_info['verbs']:
            verb = verb_info['verb']
            row_info = {'verb': verb}
            pas_parts = re.findall(r'\[[^]]*\]', verb_info['description'])
            for pas_part in pas_parts:
                pas_part = pas_part[1:-1]  # remove opening and closing brackets
                pas_part_fields = pas_part.split(": ")
                pas_key = pas_part_fields[0]
                # This is because the argument can contain the string ": ".
                pas_value = ": ".join(pas_part_fields[1:])
                if pas_key == "V":
                    continue
                if pas_key not in column_names:
                    column_names.append(pas_key)
                row_info[pas_key] = pas_value
            row_info['sentence_id'] = str(sentence_id)
            table_info.append(row_info)
    return table_info, column_names, processed_sentences


def get_tagged_info(table_info: List[Dict[str, str]],
                    processed_sentences: List[Span]) -> List[Dict[str, Dict[str, str]]]:
    tagged_info: List[Dict[str, Dict[str, str]]] = []
    for row_info in table_info:
        spacy_sentence = processed_sentences[int(row_info['sentence_id'])]
        entities_in_sentence = spacy_sentence.ents
        numbers_in_sentence = []
        dates_in_sentence = []
        other_entities_in_sentence = []
        for entity in entities_in_sentence:
            if entity.label_ == 'CARDINAL':
                numbers_in_sentence.append(entity.string)
            elif entity.label_ == 'DATE':
                dates_in_sentence.append(entity.string)
            else:
                other_entities_in_sentence.append(entity.string)
        tagged_row_info: Dict[str, Dict[str]] = {}
        for relation_name, argument in row_info.items():
            if relation_name == 'sentence_id':
                # We're not extracting lemma, numbers, dates or entities here.
                tagged_row_info[relation_name] = argument
            elif relation_name == 'verb':
                # We're only extracting the lemma here.
                tagged_row_info[relation_name] = {"string": argument,
                                                  "lemmas": [token.lemma_ for token in SPACY_NLP(argument)]}
            else:
                # We're assuming that any number, date, or entity that matches a substring in this
                # argument was extracted by Spacy from this argument. This is not always correct. For
                # example, if the sentence has the same number in multiple arguments, it shows up
                # multiple times within each argument this way. But it is not a serious problem.
                numbers_in_argument = [number.strip() for number in numbers_in_sentence if number in
                                       argument]
                dates_in_argument = [date.strip() for date in dates_in_sentence if date in
                                     argument]
                entities_in_argument = [entity.strip() for entity in other_entities_in_sentence if entity in
                                        argument]
                tagged_row_info[relation_name] = {"argument_string": argument,
                                                  'argument_lemmas': [token.lemma_ for token in
                                                                      SPACY_NLP(argument)],
                                                  "numbers": numbers_in_argument,
                                                  "dates": dates_in_argument,
                                                  "entities": entities_in_argument}
        tagged_info.append(tagged_row_info)
    return tagged_info


def make_files_for_semparse(data_file: str,
                            output_path: str,
                            tagger: Callable[[str], JsonDict]):
    data_file_suffix = data_file.split("/")[-1].replace(".json", "")
    examples_file_path = f"{output_path}/{data_file_suffix}.examples"
    tables_path = f"{output_path}/tables"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(tables_path):
        os.mkdir(tables_path)

    data = json.load(open(data_file))
    LOGGER.info(f"Read data from {data_file}")
    LOGGER.info(f"Writing examples to {examples_file_path}..")
    examples_file = open(examples_file_path, "w")
    question_counter = 0
    for paragraph_id, paragraph_data in data.items():
        passage = paragraph_data['passage']
        processed_doc = SPACY_NLP(passage)
        table_info, column_names, processed_sentences = get_table_info(processed_doc, tagger)
        tagged_info = get_tagged_info(table_info, processed_sentences)
        sentences = [spacy_sentence.string.strip() for spacy_sentence in processed_sentences]
        table_file = f"{tables_path}/{paragraph_id}.table"
        LOGGER.info(f"Writing {table_file}..")
        with open(table_file, "w") as output_file:
            print("\t".join(column_names), file=output_file)
            for row_info in table_info:
                print("\t".join([row_info[column_name] if column_name in row_info else "NULL"
                                 for column_name in column_names]), file=output_file)

        sentences_file = f"{tables_path}/{paragraph_id}.sentences"
        LOGGER.info(f"Writing {sentences_file}..")
        with open(sentences_file, "w") as output_file:
            for sentence_id, sentence in enumerate(sentences):
                print(f"{sentence_id}\t{sentence}", file=output_file)

        tagged_file = f"{tables_path}/{paragraph_id}.tagged"
        LOGGER.info(f"Writing {tagged_file}..")
        with open(tagged_file, "w") as output_file:
            tagged_header_fields = ["row", "col", "id", "content", "tokens", "lemmaTokens",
                                    "posTags", "nerTags", "nerValues", "number", "date", "num2",
                                    "list", "listId"]
            print("\t".join(tagged_header_fields), file=output_file)
            for i, column_name in enumerate(column_names):
                # Convert the column names into SEMPRE notation so that we can directly use the
                # table context code.
                normalized_name = "relation:" + column_name.lower().replace("-", "_")
                fields = ['-1', str(i), normalized_name, column_name]
                fields = fields + [''] * (len(tagged_header_fields) - 4)
                print("\t".join(fields), file=output_file)
            for row_id, row_info in enumerate(tagged_info):
                for column_id, column_name in enumerate(column_names):
                    if column_name in row_info:
                        if column_name == 'sentence_id':
                            continue
                        elif column_name == 'verb':
                            content = row_info[column_name]["string"]
                            lemma = "|".join(row_info[column_name]["lemmas"])
                            numbers, dates, entities = [], [], []
                        else:
                            content = row_info[column_name]['argument_string']
                            numbers = row_info[column_name]['numbers']
                            dates = row_info[column_name]['dates']
                            entities = row_info[column_name]['entities']
                            lemma = "|".join(row_info[column_name]["argument_lemmas"])
                    else:
                        content, lemma, numbers, dates, entities = "NULL", "NULL", [], [], []
                    normalized_name = content.lower().replace(" ", "_")
                    ner_values = "|".join(entities)  # Note that we only include NEs that are not numbers or dates.
                    number_values = "|".join(numbers)
                    date_values = "|".join(dates)
                    fields = [''] * 14
                    fields[0] = str(row_id)
                    fields[1] = str(column_id)
                    fields[2] = normalized_name
                    fields[3] = content
                    fields[5] = lemma
                    fields[8] = ner_values
                    fields[9] = number_values
                    fields[10] = date_values
                    print("\t".join(fields), file=output_file)

        for qa_pair in paragraph_data["qa_pairs"]:
            question = qa_pair["question"].lower()
            answers = []
            if qa_pair['answer']['date']:
                year, month, day = "xxxx", "xx", "xx"
                date = qa_pair["answer"]["date"]
                if "year" in date and date["year"]:
                    year = date["year"]
                if "month" in date:
                    month = MONTH_NUMBERS.get(date["month"].lower(), "xx")
                if "day" in date and date["day"]:
                    day = date["day"]
                date_string = f"{year}-{month}-{day}"
                if date_string != "xxxx-xx-xx":
                    answers.append(date_string)

            if qa_pair['answer']['number']:
                answers.append(qa_pair['answer']['number'])

            if qa_pair['answer']['spans']:
                answers.extend(qa_pair['answer']['spans'])
            question_lisp = f'(utterance "{question}")'
            answer_descriptions = " ".join([f'(description "{answer}")' for answer in answers])
            answer_lisp = f'(targetValue (list {answer_descriptions}))'
            context_lisp = f"(context (graph tables.TableKnowledgeGraph tables/{paragraph_id}.tagged))"
            example_lisp = f"(example (id q-{question_counter}) {question_lisp} {context_lisp} {answer_lisp})"
            question_counter += 1
            print(example_lisp, file=examples_file)
    examples_file.close()
    LOGGER.log(f"Wrote {question_counter} questions.")

def main(args):
    if args.tagger == "srl":
        model = srl_with_elmo_luheng_2018()
        tagger_function = model.predict
    elif args.tagger == "oie":
        model = open_information_extraction_stanovsky_2018()
        tagger_function = lambda sentence: model.predict_json({"sentence": sentence})
    else:
        raise RuntimeError(f"Unknown tagger type: {args.tagger}")
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    make_files_for_semparse(args.data_file, args.output_path, tagger_function)


if __name__ == '__main__':
    # pylint: disable=invalid-name
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_file", type=str, help="Path to input data file in JSON format")
    argparser.add_argument("output_path", type=str, help="Path to the output directory")
    argparser.add_argument("tagger", type=str, help="Type of tagger to use")
    argparser.add_argument("--verbose", help="Verbose output", action="store_true")
    arguments = argparser.parse_args()
    main(arguments)
