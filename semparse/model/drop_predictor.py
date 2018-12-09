import os

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util
from allennlp.predictors.predictor import Predictor


@Predictor.register('drop-parser')
class DropParserPredictor(Predictor):
    @overrides
    def load_line(self, line: str) -> JsonDict:
        parsed_info = wikitables_util.parse_example_line(line)

        question = parsed_info["question"]
        question_id = parsed_info["id"]
        # We want the tagged file, but the ``*.examples`` files typically point to CSV.
        # pylint: disable=protected-access
        table_filename = os.path.join(self._dataset_reader._tables_directory,
                                      parsed_info["table_filename"].replace("csv", "tagged"))

        table_lines = [line.split("\t") for line in open(table_filename).readlines()]
        return {"question_id": question_id,
                "question": question,
                "table_lines": table_lines}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(question=json_dict["question"],
                                                     table_lines=json_dict["table_lines"],
                                                     target_values=None)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        question_id = inputs["question_id"]
        instance = self._json_to_instance(inputs)
        outputs = self.predict_instance(instance)
        outputs["question_id"] = question_id
        return outputs

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        question_id = outputs["question_id"]
        denotation = outputs["denotations"]
        return f'"{question_id}": {denotation},\n'
