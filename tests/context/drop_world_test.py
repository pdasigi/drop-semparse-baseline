# pylint: disable=no-self-use
# pylint: disable=invalid-name
# pylint: disable=too-many-public-methods
from typing import List
from functools import partial

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.semparse import ParsingError

from semparse.context import ParagraphQuestionContext, DropWorld
from semparse.language import types


def check_productions_match(actual_rules: List[str], expected_right_sides: List[str]):
    actual_right_sides = [rule.split(' -> ')[1] for rule in actual_rules]
    assert set(actual_right_sides) == set(expected_right_sides)

# TODO(pradeep): Add tests!
class DropWorldTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
