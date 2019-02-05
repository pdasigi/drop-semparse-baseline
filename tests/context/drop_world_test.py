# pylint: disable=no-self-use
# pylint: disable=invalid-name
# pylint: disable=too-many-public-methods
from typing import List

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer

from semparse.context import ParagraphQuestionContext, DropWorld


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)

# TODO(pradeep): Add tests!
class DropWorldTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer()
        tokens = self.tokenizer.tokenize("""how many points did the redskins score in the final two minutes of the
                                         game?""")
        context = ParagraphQuestionContext.read_from_file("fixtures/data/tables/sample_paragraph.tagged",
                                                          tokens)
        self.world = DropWorld(context)

    def test_get_agenda(self):
        assert self.world.get_agenda() == ['<p,n> -> count_structures', 's -> string:point',
                                           's -> string:redskin', 's -> string:score', 's -> string:two',
                                           's -> string:game']
