# pylint: disable=no-self-use
# pylint: disable=invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from semparse.context import ParagraphQuestionContext


# TODO(pradeep): Add more tests
class TestParagraphQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        question = "did the redskins score in the final two minutes of the game?"
        question_tokens = self.tokenizer.tokenize(question)
        self.test_file = 'fixtures/data/sample_paragraph.tagged'
        self.context = ParagraphQuestionContext.read_from_file(self.test_file, question_tokens)

    def _get_context_with_question(self, question):
        question_tokens = self.tokenizer.tokenize(question)
        context = ParagraphQuestionContext.read_from_file(self.test_file, question_tokens)
        return context

    def test_paragraph_data(self):
        paragraph_data = self.context.paragraph_data
        verbs = [structure["relation:verb"] for structure in paragraph_data]
        verb_strings = [verb.argument_string for verb in verbs]
        verb_lemmas = [verb.argument_lemmas for verb in verbs]
        assert verb_strings == ["started", "allowed", "caused", "recovered", "were", "see", "play", "was",
                                "fumbled", "did", "score", "scored", "win", "scored", "rushing",
                                "recovered"]
        assert verb_lemmas == [["start"], ["allow"], ["cause"], ["recover"], ["be"], ["see"], ["play"], ["be"],
                               ["fumble"], ["do"], ["score"], ["score"], ["win"], ["score"],
                               ["rush"], ["recover"]]

    def test_get_entities_from_question(self):
        entities, numbers = self.context.get_entities_from_question()
        # The entities are lemmas of questions tokens that match lemmas of words in paragraphs.
        assert entities == ['string:redskin', 'string:score', 'string:two', 'string:game']
        assert numbers == [('2', 7)]
        context = self._get_context_with_question("""Were the redskins seen to be scoring in the final
                                                  two minutes?""")
        entities, numbers = context.get_entities_from_question()
        assert entities == ['string:be', 'string:redskin', 'string:see', 'string:score',
                            'string:two']
        assert numbers == [('2', 10)]
