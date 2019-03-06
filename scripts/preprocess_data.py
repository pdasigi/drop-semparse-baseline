import json
import tqdm
import os

import argparse
import logging
from copy import copy
from typing import Callable, Dict, List, Tuple, Optional

import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree

import spacy
from spacy.tokens import Span, Doc
from allennlp.models.model import Model
from allennlp.common.util import JsonDict
from allennlp.pretrained import (srl_with_elmo_luheng_2018,
                                 open_information_extraction_stanovsky_2018,
                                 biaffine_parser_stanford_dependencies_todzat_2017,
                                 neural_coreference_resolution_lee_2017)


LOGGER = logging.getLogger(__name__)

SPACY_NLP = spacy.load('en')

def format_single_verb(nx_graph, words, verb_ind):
    """
    Get a single verb description.
    """
    desc = words
    bio_tags = ["O" for _ in words]
    desc[verb_ind] = f"[V: {desc[verb_ind]}]"
    bio_tags[verb_ind] = "B-V"
    for dep in nx_graph[verb_ind]:
        label = nx_graph[verb_ind][dep]["label"]
        subtree = dfs_tree(nx_graph, dep)
        start_word_ind = min(subtree)
        start_word = desc[start_word_ind]
        end_word_ind = max(subtree)

        # Update description
        desc[start_word_ind] = f"[{label}: {start_word}"
        desc[end_word_ind] = f"{desc[end_word_ind]}]"

        # Update tags
        bio_tags[start_word_ind] = f"B-{label}"
        for cur_ind in range(start_word_ind + 1, end_word_ind + 1):
            bio_tags[cur_ind] = f"I-{label}"
    return desc, bio_tags

def get_verb_info_from_graph(nx_graph):
    """
    Convert an nx_graph to AllenNLP "description".
    {'verb': 'Hoping', 'description': '[V: Hoping] [ARG1: to get their first win of the season] ,
    the 49ers went home for a Week 5 Sunday night duel with the Philadelphia Eagles .',
    'tags': ['B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1',
             'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
    """
    items = sorted(nx_graph.nodes.items())
    verbs = [(ind, attrs["word"], attrs["pos"])
             for ind, attrs in items
             if attrs["pos"].startswith("V")]
    words = [attrs["word"]
             for _, attrs in items]
    verb_info = []

    for verb_ind, verb, _ in verbs:
        desc, tags = format_single_verb(nx_graph, copy(words), verb_ind)
        verb_info.append({"verb": verb,
                          "description": " ".join(desc[1:]),
                          "tags": tags[1:]})

    return {"verbs": verb_info,
            "words": words}

def get_nx_graph_from_dep(dep_tree):
    """
    Convert an AllenNLP dep tree representation to networkx
    representation.
    """
    nx_graph = nx.DiGraph()
    nx_graph.add_node(0, word="ROOT", pos="ROOT")
    for dep_ind, (word, pos, head_ind, label) in enumerate(zip(dep_tree["words"],
                                                               dep_tree["pos"],
                                                               dep_tree["predicted_heads"],
                                                               dep_tree["predicted_dependencies"])):
        nx_graph.add_node(dep_ind + 1, word=word, pos=pos)
        nx_graph.add_edge(head_ind, dep_ind + 1, label=label)

    return nx_graph


def get_table_info(processed_passage: Doc,
                   tagger: Callable[[str], JsonDict]) -> Tuple[List[Dict[str, str]], List[str], List[Span]]:
    processed_sentences: List[Span] = []
    table_info: List[Dict[str, str]] = []
    column_names = ['sentence_id', 'verb']
    for sentence_id, spacy_sentence in enumerate(processed_passage.sents):
        processed_sentences.append(spacy_sentence)
        pas_info = tagger(spacy_sentence.string.strip())
        for verb_info in pas_info['verbs']:
            tag_span_indices = {}
            prev_tag_type = None
            row_info = {}
            try:
                for i, tag in enumerate(verb_info['tags']):
                    if tag == "O":
                        tag_pos = "O"
                        tag_type = None
                    else:
                        tag_pos = tag[0]  # B or I
                        tag_type = tag[2:]
                    if tag_pos in ["B", "O"]:
                        if prev_tag_type is not None:
                            # previous_tag has ended
                            tag_span_indices[prev_tag_type].append(i)
                        if tag_pos == "B":
                            # New tag has begun
                            tag_span_indices[tag_type] = [i]
                    prev_tag_type = tag_type
                if tag_type is not None:
                    # The last tag was an "I-" tag
                    tag_span_indices[prev_tag_type].append(len(verb_info["tags"]))
                verb_span_begin, verb_span_end = tag_span_indices["V"]
                # Assuming tagger uses spacy tokenizer!
                row_info["verb"] = spacy_sentence[verb_span_begin:verb_span_end]
                for tag_type, tag_span in tag_span_indices.items():
                    if tag_type == "V":
                        continue
                    span_begin, span_end = tag_span
                    if tag_type not in column_names:
                        column_names.append(tag_type)
                        row_info[tag_type] = spacy_sentence[span_begin:span_end]
                row_info['sentence_id'] = str(sentence_id)
                table_info.append(row_info)
            except KeyError as error:
                LOGGER.warning(f"Exception in {verb_info}")
                LOGGER.warning(error)
                LOGGER.warning("Skipping structure")
            except ValueError as error:
                LOGGER.warning(f"Exception in {verb_info}")
                LOGGER.warning(error)
                LOGGER.warning("Skipping structure")
    return table_info, column_names, processed_sentences


def get_tagged_info(table_info: List[Dict[str, Span]],
                    processed_sentences: List[Span],
                    coref_clusters: List[List[Span]]) -> List[Dict[str, Dict[str, str]]]:
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
                tagged_row_info[relation_name] = {"string": argument.string.strip(),
                                                  "lemmas": [token.lemma_ for token in argument]}
            else:
                # We're assuming that any number, date, or entity that matches a substring in this
                # argument was extracted by Spacy from this argument. This is not always correct. For
                # example, if the sentence has the same number in multiple arguments, it shows up
                # multiple times within each argument this way. But it is not a serious problem
                numbers_in_argument = []
                dates_in_argument = []
                entities_in_argument = []
                for entity in argument.ents:
                    entity_text = entity.string.strip()
                    if entity.label_ == "CARDINAL":
                        numbers_in_argument.append(entity_text)
                    elif entity.label_ == "DATE":
                        dates_in_argument.append(entity_text)
                    else:
                        entities_in_argument.append(entity_text)

                # Adding coref information
                if coref_clusters:
                    for token in argument:
                        if token.lemma_ == '-PRON-':
                            for coref_cluster in coref_clusters:
                                # Assuming the first entity in the cluster that is not a PRON
                                # is the "head" of the cluster.
                                cluster_head = None
                                for cluster_item in coref_cluster:
                                    if cluster_item.lemma_ != "-PRON-":
                                        cluster_head = cluster_item
                                        break
                                token_in_cluster = False
                                for span in coref_cluster:
                                    if token in span:
                                        token_in_cluster = True
                                        break
                                if token_in_cluster and cluster_head is not None:
                                    entities_in_argument.insert(0, cluster_head.string.strip())

                tagged_row_info[relation_name] = {"argument_string": argument.string.strip(),
                                                  'argument_lemmas': [token.lemma_ for token in
                                                                      argument],
                                                  "numbers": numbers_in_argument,
                                                  "dates": dates_in_argument,
                                                  "entities": entities_in_argument}
        tagged_info.append(tagged_row_info)
    return tagged_info


def make_files_for_semparse(data_files_path: str,
                            output_path: str,
                            tagger: Callable[[str], JsonDict],
                            coref_model: Optional[Model]):
    tables_path = f"{output_path}/tables"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(tables_path):
        os.mkdir(tables_path)

    data = {}
    for file_name in os.listdir(data_files_path):
        if file_name.endswith('.json'):
            data_file = os.path.join(data_files_path, file_name)
            data.update(json.load(open(data_file)))
            LOGGER.info(f"Read data from {data_file}")
    paragraph_counter = 0
    for paragraph_id, paragraph_data in tqdm.tqdm(data.items()):
        passage = paragraph_data['passage']
        processed_doc = SPACY_NLP(passage)
        coref_spans: List[List[Span]] = []
        if coref_model is not None:
            coref_prediction = coref_model.predict(passage)
            coref_clusters = coref_prediction['clusters']
            for cluster in coref_clusters:
                coref_spans.append([])
                for span_begin, span_end in cluster:
                    coref_spans[-1].append(processed_doc[span_begin:span_end + 1])
        table_info, column_names, processed_sentences = get_table_info(processed_doc, tagger)
        tagged_info = get_tagged_info(table_info, processed_sentences, coref_spans)
        sentences = [spacy_sentence.string.strip() for spacy_sentence in processed_sentences]
        table_file = f"{tables_path}/{paragraph_id}.table"
        LOGGER.info(f"Writing {table_file}..")
        with open(table_file, "w") as output_file:
            print("\t".join(column_names), file=output_file)
            for row_info in table_info:
                row_info_to_print = []
                for column_name in column_names:
                    if column_name in row_info:
                        if isinstance(row_info[column_name], Span):
                            column_info = row_info[column_name].string.strip()
                        else:
                            column_info = row_info[column_name]
                        row_info_to_print.append(column_info)
                    else:
                        row_info_to_print.append('NULL')
                print('\t'.join(row_info_to_print), file=output_file)

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
        paragraph_counter += 1

    LOGGER.info(f"Wrote {paragraph_counter} paragraphs.")


def main(args):
    if args.tagger == "srl":
        model = srl_with_elmo_luheng_2018()
        tagger_function = model.predict
    elif args.tagger == "oie":
        model = open_information_extraction_stanovsky_2018()
        tagger_function = lambda sentence: model.predict_json({"sentence": sentence})
    elif args.tagger == "dep":
        model = biaffine_parser_stanford_dependencies_todzat_2017()
        tagger_function = lambda sentence: get_verb_info_from_graph(get_nx_graph_from_dep(model.predict(sentence)))
    else:
        raise RuntimeError(f"Unknown tagger type: {args.tagger}")
    if args.include_coref:
        coref_model = neural_coreference_resolution_lee_2017()
    else:
        coref_model = None
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    make_files_for_semparse(args.data_files_path, args.output_path, tagger_function, coref_model)


if __name__ == '__main__':
    # pylint: disable=invalid-name
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_files_path", type=str,
                           help="Path to directory containing input data file(s) in JSON format")
    argparser.add_argument("output_path", type=str, help="Path to the output directory")
    argparser.add_argument("tagger", type=str, help="Type of tagger to use")
    argparser.add_argument("--include-coref", dest="include_coref", action="store_true",
                           help="Use a coref system to include resolved entities")
    argparser.add_argument("--verbose", help="Verbose output", action="store_true")
    arguments = argparser.parse_args()
    main(arguments)
