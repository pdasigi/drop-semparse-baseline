import json
import logging

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


LOGGER = logging.getLogger(__name__)


@Predictor.register('drop-parser')
class DropParserPredictor(Predictor):
    @overrides
    def load_line(self, line: str) -> JsonDict:
        try:
            parsed_info = json.loads(line)

            question = parsed_info["question"]
            question_id = parsed_info["question_id"]
            table_lines = parsed_info["table_lines"]
            return {"question_id": question_id,
                    "question": question,
                    "table_lines": table_lines}
        except json.decoder.JSONDecodeError as error:
            LOGGER.error("""Encountered an unexpected data format. If you see this error while running the
                         predictor from command line, you probably did not preprocess the data into jsonl format.
                         See ``scripts/make_json_lines_data.py``.""")
            raise error

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(question=json_dict["question"],
                                                     table_lines=json_dict["table_lines"],
                                                     answer_json=None)

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
        denotation = "\t".join(outputs["denotations"])
        logical_form = outputs["logical_form"][0]
        return f"{question_id}\t{logical_form}\t{denotation}\n"
