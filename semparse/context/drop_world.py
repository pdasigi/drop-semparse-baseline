"""
We store the information related to context sensitive execution of logical forms here.
We assume that the logical forms are written in the variable-free language defined by
``semparse.drop_type_declaration``
"""
from typing import Dict, List, Set, Union
import re
import logging

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World

from semparse.context.paragraph_question_context import ParagraphQuestionContext
from semparse.language import drop_type_declaration as types
from semparse.language.drop_executor import DropExecutor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DropWorld(World):
    """
    Parameters
    ----------
    paragraph_context : ``ParagraphQuestionContext``
        Context associated with this world.
    """
    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = {
            types.SELECT_TYPE: 2,
            types.PAS_FILTER_WITH_RELATION: 2,
            types.PAS_NUM_OP: 2,
            types.NUM_DIFF_TYPE: 2,
            types.PAS_FILTER_WITH_RELATION_AND_NUMBER: 3,
            types.PAS_FILTER_WITH_RELATION_AND_DATE: 3,
            types.PAS_FILTER_WITH_RELATION_AND_STRING: 3,
            types.DATE_FUNCTION_TYPE: 4,
            }

    def __init__(self, paragraph_context: ParagraphQuestionContext) -> None:
        super().__init__(constant_type_prefixes={"string": types.STRING_TYPE,
                                                 "num": types.NUMBER_TYPE},
                         global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                         global_name_mapping=types.COMMON_NAME_MAPPING)
        self.paragraph_context = paragraph_context

        #self.table_graph = paragraph_context.get_table_knowledge_graph()

        self._executor = DropExecutor(self.paragraph_context.paragraph_data)

        # For every new column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # Adding entities and numbers seen in questions to the mapping.
        question_entities, question_numbers = paragraph_context.get_entities_from_question()
        self._question_entities = question_entities
        self._question_numbers = [number for number, _ in question_numbers]
        for entity in self._question_entities:
            # These entities all have prefix "string:"
            self._map_name(entity, keep_mapping=True)

        for number_in_question in self._question_numbers:
            self._map_name(f"num:{number_in_question}", keep_mapping=True)

        for relation_name in paragraph_context.paragraph_data[0].keys():
            self._map_name(relation_name, keep_mapping=True)

        self.terminal_productions: Dict[str, str] = {}
        name_mapping = [(name, mapping) for name, mapping in self.global_name_mapping.items()]
        name_mapping += [(name, mapping) for name, mapping in self.local_name_mapping.items()]
        signatures = self.global_type_signatures.copy()
        signatures.update(self.local_type_signatures)
        for predicate, mapped_name in name_mapping:
            if mapped_name in signatures:
                signature = signatures[mapped_name]
                self.terminal_productions[predicate] = f"{signature} -> {predicate}"

        # We don't need to recompute this ever; let's just compute it once and cache it.
        self._valid_actions: Dict[str, List[str]] = None

    @staticmethod
    def is_instance_specific_entity(entity_name: str) -> bool:
        """
        Instance specific entities are column names, strings and numbers. Returns True if the entity
        is one of those.
        """
        entity_is_number = False
        try:
            float(entity_name)
            entity_is_number = True
        except ValueError:
            pass
        # Column names start with "*_column:", strings start with "string:"
        return "relation:" in entity_name or entity_name.startswith("string:") or entity_is_number

    @overrides
    def _get_curried_functions(self) -> Dict[Type, int]:
        return DropWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        basic_types = set(types.BASIC_TYPES)
        return basic_types

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.STARTING_TYPES

    def _translate_name_and_add_mapping(self, name: str) -> str:
        if "relation:" in name:
            # Column name
            translated_name = "C%d" % self._column_counter
            self._column_counter += 1
            self._add_name_mapping(name, translated_name, types.RELATION_TYPE)
        elif name.startswith("string:"):
            # We do not need to translate these names.
            translated_name = name
            self._add_name_mapping(name, translated_name, types.STRING_TYPE)
        elif name.startswith("num:"):
            # NLTK throws an error if it sees a "." in constants, which will most likely happen
            # within numbers as a decimal point. We're changing those to underscores.
            translated_name = name.replace(".", "_")
            if re.match("num:-[0-9_]+", translated_name):
                # The string is a negative number. This makes NLTK interpret this as a negated
                # expression and force its type to be TRUTH_VALUE (t).
                translated_name = translated_name.replace("-", "~")
            original_name = name.replace("num:", "")
            self._add_name_mapping(original_name, translated_name, types.NUMBER_TYPE)
        return translated_name

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            translated_name = self._translate_name_and_add_mapping(name)
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

    def get_agenda(self):
        """
        Get the agenda that a logical form translating this question should satisfy.
        """
        raise NotImplementedError

    def execute(self, logical_form: str) -> Union[List[str], int]:
        """Execute the logical form"""
        return self._executor.execute(logical_form)

    def evaluate_logical_form(self, logical_form: str, target_list: List[str]) -> bool:
        """
        Takes a logical forms and a list of target values as strings from the original lisp
        representation of instances, and returns True iff the logical form executes to those values.
        """
        return self._executor.evaluate_logical_form(logical_form, target_list)
