# pylint: disable=invalid-name,no-self-use,protected-access
from flaky import flaky

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.iterators import DataIterator

from semparse import *

class DropMmlParserTest(ModelTestCase):
    def setUp(self):
        super(DropMmlParserTest, self).setUp()
        config_path = "fixtures/model/experiment.json"
        data_path = "fixtures/data/sample_data.json"
        self.set_up_model(config_path, data_path)

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_model_decode(self):
        params = Params.from_file(self.param_file)
        iterator_params = params['iterator']
        iterator = DataIterator.from_params(iterator_params)
        iterator.index_with(self.model.vocab)
        model_batch = next(iterator(self.dataset, shuffle=False))
        self.model.training = False
        forward_output = self.model(**model_batch)
        decode_output = self.model.decode(forward_output)
        assert "predicted_actions" in decode_output
