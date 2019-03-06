import json
import argparse


def main(args: argparse.Namespace) -> None:
    data = json.load(open(args.data_file))
    with open(args.output_file_name, "w") as output_file:
        for passage_id, passage_info in data.items():
            table_file = f"{args.tables_directory}/{passage_id}.tagged"
            table_lines = [line.split("\t") for line in open(table_file).readlines()]
            for qa_info in passage_info["qa_pairs"]:
                question_id = qa_info["query_id"]
                question = qa_info["question"]
                answer = qa_info["answer"]
                data_line = json.dumps({"question_id": question_id,
                                        "question": question,
                                        "table_lines": table_lines,
                                        "answer": answer})
                print(data_line, file=output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Transforms data from official DROP format to a bulkier jsonl format")
    argparser.add_argument("data_file", type=str, help="Input file in official DROP data format")
    argparser.add_argument("tables_directory", type=str, help="Path to the directory containing processed tables")
    argparser.add_argument("output_file_name", type=str, help="Name of the output JSONL file")
    args = argparser.parse_args()
    main(args)
