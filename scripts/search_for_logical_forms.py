#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position
import sys
import os
import argparse
import gzip
import logging
import math
from multiprocessing import Process

from tqdm import tqdm
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from semparse.action_space_walker import ActionSpaceWalker
from semparse.context import util as context_util
from semparse.context.paragraph_question_context import ParagraphQuestionContext
from semparse.context.drop_world import DropWorld

def search(tables_directory: str,
           data: JsonDict,
           output_path: str,
           max_path_length: int,
           max_num_logical_forms: int,
           use_agenda: bool,
           output_separate_files: bool,
           embedding_file: str,
           distance_threshold: float) -> None:
    print(f"Starting search with {len(data)} instances", file=sys.stderr)
    executor_logger = logging.getLogger('weak_supervision.executors.drop_executor')
    executor_logger.setLevel(logging.ERROR)
    tokenizer = WordTokenizer()
    embedding = None
    if embedding_file:
        print("Reading from pretrained embedding file.")
        embedding = context_util.read_pretrained_embedding(embedding_file)
    if output_separate_files and not os.path.exists(output_path):
        os.makedirs(output_path)
    if not output_separate_files:
        output_file_pointer = open(output_path, "w")
    for instance_data in tqdm(data):
        utterance = instance_data["question"]
        question_id = instance_data["id"]
        print("Processing", question_id)
        if utterance.startswith('"') and utterance.endswith('"'):
            utterance = utterance[1:-1]
        # For example: csv/200-csv/47.csv -> tagged/200-tagged/47.tagged
        table_file = instance_data["table_filename"].replace("csv", "tagged")
        target_list = instance_data["target_values"]
        tokenized_question = tokenizer.tokenize(utterance)
        table_file = f"{tables_directory}/{table_file}"
        context = ParagraphQuestionContext.read_from_file(table_file,
                                                          tokenized_question,
                                                          embedding,
                                                          distance_threshold)
        world = DropWorld(context)
        walker = ActionSpaceWalker(world, max_path_length=max_path_length)
        correct_logical_forms = []
        if use_agenda:
            agenda = world.get_agenda()
            all_logical_forms = walker.get_logical_forms_with_agenda(agenda=agenda,
                                                                     max_num_logical_forms=10000,
                                                                     allow_partial_match=True)
        else:
            all_logical_forms = walker.get_all_logical_forms(max_num_logical_forms=10000)
        for logical_form in all_logical_forms:
            if world.evaluate_logical_form(logical_form, target_list):
                correct_logical_forms.append(logical_form)
        if output_separate_files and correct_logical_forms:
            with gzip.open(f"{output_path}/{question_id}.gz", "wt") as output_file_pointer:
                for logical_form in correct_logical_forms:
                    print(logical_form, file=output_file_pointer)
        elif not output_separate_files:
            print(f"{question_id} {utterance}", file=output_file_pointer)
            if use_agenda:
                print(f"Agenda: {agenda}", file=output_file_pointer)
            if not correct_logical_forms:
                print("NO LOGICAL FORMS FOUND!", file=output_file_pointer)
            for logical_form in correct_logical_forms[:max_num_logical_forms]:
                print(logical_form, file=output_file_pointer)
            print(file=output_file_pointer)
    if not output_separate_files:
        output_file_pointer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_directory", type=str, help="Location of the tables")
    parser.add_argument("data_file", type=str, help="Path to the *.examples file")
    parser.add_argument("output_path", type=str, help="""Path to the output directory if
                        'output_separate_files' is set, or to the output file if not.""")
    parser.add_argument("--max-path-length", type=int, dest="max_path_length", default=10,
                        help="Max length to which we will search exhaustively")
    parser.add_argument("--max-num-logical-forms", type=int, dest="max_num_logical_forms",
                        default=100, help="Maximum number of logical forms returned")
    parser.add_argument("--use-agenda", dest="use_agenda", action="store_true",
                        help="Use agenda to sort the output logical forms")
    parser.add_argument("--output-separate-files", dest="output_separate_files",
                        action="store_true", help="""If set, the script will output gzipped
                        files, one per example. You may want to do this if you;re making data to
                        train a parser.""")
    parser.add_argument("--num-splits", dest="num_splits", type=int, default=0,
                        help="Number of splits to make of the data, to run as many processes (default 0)")
    parser.add_argument("--embedding-file", dest="embedding_file", type=str,
                        help="""If provided, this file will be used to extract paragraph tokens that
                        are similar to question tokens and add them to context.""")
    parser.add_argument("--distance-threshold", dest="distance_threshold", type=float, default=0.3,
                        help="Threshold to use to extract similar tokens for paragraph as entities")
    args = parser.parse_args()
    input_data = [wikitables_util.parse_example_line(example_line) for example_line in
                  open(args.data_file)]
    if args.num_splits == 0 or len(input_data) <= args.num_splits or not args.output_separate_files:
        search(args.table_directory, input_data, args.output_path, args.max_path_length,
               args.max_num_logical_forms, args.use_agenda, args.output_separate_files,
               args.embedding_file, args.distance_threshold)
    else:
        chunk_size = math.ceil(len(input_data)/args.num_splits)
        start_index = 0
        for i in range(args.num_splits):
            if i == args.num_splits - 1:
                data_split = input_data[start_index:]
            else:
                data_split = input_data[start_index:start_index + chunk_size]
            start_index += chunk_size
            process = Process(target=search, args=(args.domain, args.table_directory, data_split,
                                                   args.output_path, args.max_path_length,
                                                   args.max_num_logical_forms, args.use_agenda,
                                                   args.output_separate_files, args.embedding_file,
                                                   args.distance_threshold))
            print(f"Starting process {i}", file=sys.stderr)
            process.start()
