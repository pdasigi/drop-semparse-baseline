# pylint: disable=no-self-use
# pylint: disable=invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from semparse.context import ParagraphQuestionContext
from semparse.context import util as context_util


# TODO(pradeep): Add more tests
class TestParagraphQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        question = "did the redskins score in the final two minutes of the game?"
        question_tokens = self.tokenizer.tokenize(question)
        self.test_file = 'fixtures/data/tables/sample_paragraph.tagged'
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
        assert entities == [('string:redskin', 'relation:arg0'),
                            ('string:score', 'relation:verb'),
                            ('string:two', 'relation:arg1'),
                            ('string:game', 'relation:argm_prp')]
        assert numbers == [('2', 7)]
        context = self._get_context_with_question("""Were the redskins seen to be scoring in the final
                                                  two minutes?""")
        entities, numbers = context.get_entities_from_question()
        assert entities == [('string:be', 'relation:verb'),
                            ('string:redskin', 'relation:arg0'),
                            ('string:see', 'relation:arg2'),
                            ('string:score', 'relation:verb'),
                            ('string:two', 'relation:arg1')]
        assert numbers == [('2', 10)]

    def test_context_with_embedding_to_select_entities(self):
        question = "what resulted in the redskins not scoring in the final two minutes of the game?"
        question_tokens = self.tokenizer.tokenize(question)
        embedding_file = "fixtures/data/glove_100d_sample.txt.gz"
        embedding = context_util.read_pretrained_embedding(embedding_file)
        context = ParagraphQuestionContext.read_from_file(self.test_file,
                                                          question_tokens,
                                                          embedding)
        entities = context.paragraph_tokens_to_keep
        expected_entities = [('first', ['relation:arg1']),
                             ('four', ['relation:arg1']),
                             ('six', ['relation:arg1'])]
        assert all([entity in entities for entity in expected_entities])

    def test_get_knowledge_graph(self):
        knowledge_graph = self.context.get_knowledge_graph()
        assert knowledge_graph.entities == ['-1',
                                            '2',
                                            'relation:arg0',
                                            'relation:arg1',
                                            'relation:arg2',
                                            'relation:argm_loc',
                                            'relation:argm_mnr',
                                            'relation:argm_neg',
                                            'relation:argm_prp',
                                            'relation:argm_tmp',
                                            'relation:verb',
                                            'string:game',
                                            'string:redskin',
                                            'string:score',
                                            'string:two']
        assert knowledge_graph.neighbors == {'relation:verb': ['string:score'],
                                             'relation:arg0': ['string:redskin'],
                                             'relation:arg1': ['string:two'],
                                             'relation:argm_loc': [],
                                             'relation:argm_tmp': [],
                                             'relation:arg2': [],
                                             'relation:argm_neg': [],
                                             'relation:argm_prp': ['string:game'],
                                             'relation:argm_mnr': [],
                                             'string:redskin': ['relation:arg0'],
                                             'string:score': ['relation:verb'],
                                             'string:two': ['relation:arg1'],
                                             'string:game': ['relation:argm_prp'],
                                             '2': [],
                                             '-1': []}
        assert knowledge_graph.entity_text == {'relation:verb': 'verb',
                                               'relation:arg0': 'arg0',
                                               'relation:arg1': 'arg1',
                                               'relation:argm_loc': 'argm loc',
                                               'relation:argm_tmp': 'argm tmp',
                                               'relation:arg2': 'arg2',
                                               'relation:argm_neg': 'argm neg',
                                               'relation:argm_prp': 'argm prp',
                                               'relation:argm_mnr': 'argm mnr',
                                               'string:redskin': 'redskin',
                                               'string:score': 'score',
                                               'string:two': 'two',
                                               'string:game': 'game',
                                               '2': '2',
                                               '-1': '-1'}

    def test_get_knowledge_graph_with_empty_paragraph(self):
        question = "did the redskins score in the final two minutes of the game?"
        question_tokens = self.tokenizer.tokenize(question)
        empty_test_file = 'fixtures/data/tables/empty_paragraph.tagged'
        empty_context = ParagraphQuestionContext.read_from_file(empty_test_file, question_tokens)
        knowledge_graph = empty_context.get_knowledge_graph()
        assert knowledge_graph.entities == []
        assert knowledge_graph.neighbors == {}
        assert knowledge_graph.entity_text == {}
