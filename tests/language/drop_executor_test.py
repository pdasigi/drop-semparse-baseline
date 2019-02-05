# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from semparse.context import ParagraphQuestionContext
from semparse.language import DropExecutor


# TODO(pradeep): Add more tests
class DropExecutorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        question = "how many touchdowns did the redskins score??"
        question_tokens = self.tokenizer.tokenize(question)
        self.test_file = 'fixtures/data/tables/sample_paragraph.tagged'
        context = ParagraphQuestionContext.read_from_file(self.test_file, question_tokens)
        self.executor = DropExecutor(context.paragraph_data)

    def test_filter_in(self):
        logical_form = """(extract_number
                            (filter_in
                              (filter_in
                                (filter_in all_structures relation:verb string:score)
                               relation:arg0 string:redskin)
                             relation:arg1 string:touchdown)
                           relation:arg1)"""
        result = self.executor.execute(logical_form)
        assert result == 2

    def test_filter_not_in(self):
        # Counting the number of touchdowns not scored by redskins.
        logical_form = """(count_structures
                            (filter_in
                              (filter_not_in
                                (filter_in all_structures relation:verb string:score)
                               relation:arg0 string:redskin)
                             relation:arg1 string:touchdown))"""
        result = self.executor.execute(logical_form)
        assert result == 2
