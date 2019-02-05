import sys
import os
from typing import List, Dict, Union, Tuple, Any
from collections import defaultdict
import logging

from allennlp.common import JsonDict
from allennlp.semparse import util as semparse_util
from allennlp.semparse.domain_languages.domain_language import ExecutionError

from semparse.context.paragraph_question_context import Argument, Date

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))
sys.path.insert(0, os.path.join(ROOT_PATH, 'evaluation'))

import evaluate as evaluator  # pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)

NestedList = List[Union[str, List]]  # pylint: disable=invalid-name


class DropExecutor:
    # pylint: disable=too-many-public-methods
    """
    Implements the functions in the variable free language we use for discrete reasoning over
    paragraphs.

    Parameters
    ----------
    paragraph_data : ``List[Dict[str, ArgumentData]]``
        All the predicate-argument structures in the paragraph on which the executor will be used.
        The class expects each structure to be represented as a dict from relation names to
        corresponding ``ArgumentData``.
    """
    def __init__(self,
                 paragraph_data: List[Dict[str, Argument]],
                 verbose: bool = False) -> None:
        self.paragraph_data = paragraph_data
        self.verbose = verbose

    def __eq__(self, other):
        if not isinstance(other, DropExecutor):
            return False
        return self.paragraph_data == other.paragraph_data

    def execute(self, logical_form: str) -> Any:
        if not logical_form.startswith("("):
            logical_form = f"({logical_form})"
        logical_form = logical_form.replace(",", " ")
        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)
        result = self._handle_expression(expression_as_list)
        return result

    def evaluate_logical_form(self, logical_form: str, answer_json: JsonDict) -> Tuple[float, float]:
        """
        Takes a logical form, and the answer dict from the original dataset, and returns exact match
        and f1 measures, according to the two official metrics.
        """
        answer_string, _ = evaluator.to_string(answer_json)
        if not self.verbose:
            executor_logger = logging.getLogger('semparse.language.drop_executor')
            executor_logger.setLevel(logging.ERROR)
        try:
            denotation = self.execute(logical_form)
        except Exception:  # pylint: disable=broad-except
            if self.verbose:
                LOGGER.warning(f'Failed to execute: {logical_form}')
            return 0.0, 0.0
        if isinstance(denotation, list):
            denotation_list = [str(denotation_item) for denotation_item in denotation]
        else:
            denotation_list = [str(denotation)]
        em_score, f1_score = evaluator.get_metrics(denotation_list, answer_string)
        return em_score, f1_score

    ## Helper functions
    def _handle_expression(self, expression_list):
        if isinstance(expression_list, list) and len(expression_list) == 1:
            expression = expression_list[0]
        else:
            expression = expression_list
        if isinstance(expression, list):
            # This is a function application.
            function_name = expression[0]
        else:
            # This is a constant (like "all_structures" or "2005")
            return self._handle_constant(expression)
        try:
            function = getattr(self, function_name)
            return function(*expression[1:])
        except AttributeError:
            raise ExecutionError(f"Function not found: {function_name}")

    def _handle_constant(self, constant: str) -> Union[List[Dict[str, Argument]], str, float]:
        if constant == "all_structures":
            return self.paragraph_data
        try:
            return float(constant)
        except ValueError:
            # The constant is not a number. Returning as-is if it is a string.
            if constant.startswith("string:"):
                return constant.replace("string:", "")
            raise ExecutionError(f"Cannot handle constant: {constant}")

    def _get_structure_index(self, structure: Dict[str, Argument]) -> int:
        """
        Takes a structure and returns its index in the full list of structure.
        """
        structure_index = -1
        for index, paragraph_structure in enumerate(self.paragraph_data):
            if paragraph_structure == structure:
                structure_index = index
                break
        return structure_index

    ## Functions in the language
    def select(self, structure_expression_list: NestedList, relation_name: str) -> List[str]:
        """
        Select function takes a list of structures and a relation name and returns a list of
        argument values as strings.
        """
        structure_list = self._handle_expression(structure_expression_list)
        return [structure[relation_name].argument_string for structure in structure_list]

    def extract_entity(self, structure_expression_list: NestedList, relation_name: str) -> List[str]:
        """
        Takes a list of structures and a relation name and returns a list of entities under the
        given relation. We will select the first entity if there are multiple entities in a given
        argument.
        """
        structure_list = self._handle_expression(structure_expression_list)
        entities_to_return = []
        for structure in structure_list:
            if structure[relation_name].entities:
                entities_to_return.append(structure[relation_name].entities[0])
        return entities_to_return

    def extract_number(self, structure_expression_list: NestedList, relation_name: str) -> float:
        """
        Takes a structure (as a list to be consistent with the rest of the API), and a relation name
        and returns the number under the given relation. We will select the first number if there
        are multiple numbers in a given argument, and the first structure if the expression list
        evaluates to multiple structures. If there are no numbers in the argument, or if the
        expression does not evaluate to any structures, we will return -1.0. We do not throw an
        error instead because the grammar does not distinguish arguments with numbers from those
        without.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return -1.0
        if not structure_list[0][relation_name].numbers:
            return -1.0
        return float(structure_list[0][relation_name].numbers[0])

    def count_entities(self, structure_expression_list: NestedList, relation_name: str) -> float:
        """
        Takes a structure (as a list to be consistent with the rest of the API), and a relation name
        and returns the number of entities in that structure under the given relation. We will
        select the first structure if the expression list evaluates to multile structures.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return 0.0
        return float(len(structure_list[0][relation_name].entities))

    def argmax(self, structure_expression_list: NestedList, relation_name: str) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures and a relation name and returns a list containing a single structure (dict from
        relations to arguments) that has the maximum numerical value in the given relation. We return a list
        instead of a single dict to be consistent with the return type of `select` and `all_structures`.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        num_arguments_with_numbers = sum([structure[relation_name].numbers != [] for structure in
                                          structure_list])
        num_arguments_with_dates = sum([structure[relation_name].dates != [] for structure in
                                        structure_list])
        if num_arguments_with_dates > num_arguments_with_numbers:
            date_structure_pairs_to_compare = [(structure[relation_name].dates[0], structure) for
                                               structure in structure_list if
                                               structure[relation_name].dates]
            if not date_structure_pairs_to_compare:
                return []
            return [sorted(date_structure_pairs_to_compare, key=lambda x: x[0], reverse=True)[0][1]]
        number_structure_pairs_to_compare = [(structure[relation_name].numbers[0], structure) for
                                             structure in structure_list if
                                             structure[relation_name].numbers]
        if not number_structure_pairs_to_compare:
            return []
        return [sorted(number_structure_pairs_to_compare, key=lambda x: x[0], reverse=True)[0][1]]

    def argmin(self, structure_expression_list: NestedList, relation_name: str) -> List[Dict[str, str]]:
        """
        Takes a list of structures and a relation name and returns a list containing a single structure (dict from
        relations to arguments) that has the minimum numerical value in the given relation. We return a list
        instead of a single dict to be consistent with the return type of `select` and `all_structures`.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        num_arguments_with_numbers = sum([structure[relation_name].numbers != [] for structure in
                                          structure_list])
        num_arguments_with_dates = sum([structure[relation_name].dates != [] for structure in
                                        structure_list])
        if num_arguments_with_dates > num_arguments_with_numbers:
            date_structure_pairs_to_compare = [(structure[relation_name].dates[0], structure) for
                                               structure in structure_list if
                                               structure[relation_name].dates]
            if not date_structure_pairs_to_compare:
                return []
            return [sorted(date_structure_pairs_to_compare, key=lambda x: x[0])[0][1]]
        number_structure_pairs_to_compare = [(structure[relation_name].numbers[0], structure) for
                                             structure in structure_list if
                                             structure[relation_name].numbers]
        if not number_structure_pairs_to_compare:
            return []
        return [sorted(number_structure_pairs_to_compare, key=lambda x: x[0])[0][1]]

    def filter_number_greater(self,
                              structure_expression_list: NestedList,
                              relation_name: str,
                              value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number > filter_value:
                return_list.append(structure)
        return return_list

    def filter_number_greater_equals(self,
                                     structure_expression_list: NestedList,
                                     relation_name: str,
                                     value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number >= filter_value:
                return_list.append(structure)
        return return_list

    def filter_number_lesser(self,
                             structure_expression_list: NestedList,
                             relation_name: str,
                             value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number < filter_value:
                return_list.append(structure)
        return return_list

    def filter_number_lesser_equals(self,
                                    structure_expression_list: NestedList,
                                    relation_name: str,
                                    value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number <= filter_value:
                return_list.append(structure)
        return return_list

    def filter_number_equals(self,
                             structure_expression_list: NestedList,
                             relation_name: str,
                             value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number == filter_value:
                return_list.append(structure)
        return return_list

    def filter_number_not_equals(self,
                                 structure_expression_list: NestedList,
                                 relation_name: str,
                                 value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the number in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, float):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for number, structure in number_structure_pairs:
            if number != filter_value:
                return_list.append(structure)
        return return_list

    # Note that the following six methods are identical to the ones above, except that the filter
    # values are dates.
    def filter_date_greater(self,
                            structure_expression_list: NestedList,
                            relation_name: str,
                            value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date > filter_value:
                return_list.append(structure)
        return return_list

    def filter_date_greater_equals(self,
                                   structure_expression_list: NestedList,
                                   relation_name: str,
                                   value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date >= filter_value:
                return_list.append(structure)
        return return_list

    def filter_date_lesser(self,
                           structure_expression_list: NestedList,
                           relation_name: str,
                           value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date < filter_value:
                return_list.append(structure)
        return return_list

    def filter_date_lesser_equals(self,
                                  structure_expression_list: NestedList,
                                  relation_name: str,
                                  value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date <= filter_value:
                return_list.append(structure)
        return return_list

    def filter_date_equals(self,
                           structure_expression_list: NestedList,
                           relation_name: str,
                           value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date == filter_value:
                return_list.append(structure)
        return return_list

    def filter_date_not_equals(self,
                               structure_expression_list: NestedList,
                               relation_name: str,
                               value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures as an expression, a relation, and a numerical value expression and
        returns all the structures where the date in the corresponding argument satisfies the
        filtering condition.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        date_structure_pairs = [(structure[relation_name].dates[0], structure) for
                                structure in structure_list if
                                structure[relation_name].dates]
        filter_value = self._handle_expression(value_expression)
        if not isinstance(filter_value, Date):
            raise ExecutionError(f"Invalid filter value: {value_expression}")
        return_list = []
        for date, structure in date_structure_pairs:
            if date != filter_value:
                return_list.append(structure)
        return return_list

    def filter_in(self,
                  structure_expression_list: NestedList,
                  relation_name: str,
                  value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures, a relation, and a string value and returns all the structures where the value
        in that relation contains the given string.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        expression_evaluation = self._handle_expression(value_expression)
        if isinstance(expression_evaluation, list):
            if not expression_evaluation:
                return []
            filter_value = expression_evaluation[0]
        elif isinstance(expression_evaluation, str):
            filter_value = expression_evaluation
        else:
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        if not isinstance(filter_value, str):
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        # Assuming filter value has underscores for spaces, and are already lemmatized.
        filter_lemmas = filter_value.split("_")
        result_list = []
        for structure in structure_list:
            # Argument strings also have underscores for spaces.
            if filter_value in structure[relation_name].argument_string:
                result_list.append(structure)
            elif all([lemma in structure[relation_name].argument_lemmas for lemma in
                      filter_lemmas]):
                result_list.append(structure)
        return result_list

    def filter_not_in(self,
                      structure_expression_list: NestedList,
                      relation_name: str,
                      value_expression: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes a list of structures, a relation, and a string value and returns all the structures where the value
        in that relation does not contain the given string.
        """
        structure_list = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        expression_evaluation = self._handle_expression(value_expression)
        if isinstance(expression_evaluation, list):
            if not expression_evaluation:
                return []
            filter_value = expression_evaluation[0]
        elif isinstance(expression_evaluation, str):
            filter_value = expression_evaluation
        else:
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        if not isinstance(filter_value, str):
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        # Assuming filter value has underscores for spaces.
        filter_lemmas = filter_value.split("_")
        result_list = []
        for structure in structure_list:
            # Argument strings also have underscores for spaces.
            if filter_value not in structure[relation_name].argument_string and \
               not all([lemma in structure[relation_name].argument_lemmas for lemma in
                        filter_lemmas]):
                result_list.append(structure)
        return result_list

    def first(self, structure_expression_list: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes an expression that evaluates to a list of structures, and returns the first one in that
        list.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        if not structure_list:
            LOGGER.warning("Trying to get first structure from an empty list: %s", structure_expression_list)
            return []
        return [structure_list[0]]

    def last(self, structure_expression_list: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes an expression that evaluates to a list of structures, and returns the last one in that
        list.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        if not structure_list:
            LOGGER.warning("Trying to get last structure from an empty list: %s", structure_expression_list)
            return []
        return [structure_list[-1]]

    def previous(self, structure_expression_list: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes an expression that evaluates to a single structure, and returns the structure (as a list to be
        consistent with the rest of the API), that occurs before the input structure in the original set
        of structures. If the input structure happens to be the top structure, we will return an empty list.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        if not structure_list:
            LOGGER.warning("Trying to get the previous structure from an empty list: %s",
                           structure_expression_list)
            return []
        if len(structure_list) > 1:
            LOGGER.warning("Trying to get the previous structure from a non-singleton list: %s",
                           structure_expression_list)
        input_structure_index = self._get_structure_index(structure_list[0])  # Take the first structure.
        if input_structure_index > 0:
            return [self.paragraph_data[input_structure_index - 1]]
        return []

    def next(self, structure_expression_list: NestedList) -> List[Dict[str, Argument]]:
        """
        Takes an expression that evaluates to a single structure, and returns the structure (as a list to be
        consistent with the rest of the API), that occurs after the input structure in the original set
        of structures. If the input structure happens to be the last structure, we will return an empty list.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        if not structure_list:
            LOGGER.warning("Trying to get the next structure from an empty list: %s", structure_expression_list)
            return []
        if len(structure_list) > 1:
            LOGGER.warning("Trying to get the next structure from a non-singleton list: %s",
                           structure_expression_list)
        input_structure_index = self._get_structure_index(structure_list[-1])  # Take the last structure.
        if input_structure_index < len(self.paragraph_data) - 1 and input_structure_index != -1:
            return [self.paragraph_data[input_structure_index + 1]]
        return []

    def count_structures(self, structure_expression_list: NestedList) -> float:
        """
        Takes an expression that evaluates to a list of structures and returns their count (as a float
        to be consistent with the other functions like max that also return numbers).
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        return float(len(structure_list))

    def max(self,
            structure_expression_list: NestedList,
            relation_name: str) -> float:
        """
        Takes an expression list that evaluates to a list of structures and a relation name, and returns the max
        of the values under that relation in those structures.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        if not number_structure_pairs:
            return 0.0
        return max([value for value, _ in number_structure_pairs])

    def min(self,
            structure_expression_list: NestedList,
            relation_name: str) -> float:
        """
        Takes an expression list that evaluates to a list of structures and a relation name, and returns the min
        of the values under that relation in those structures.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        if not number_structure_pairs:
            return 0.0
        return min([value for value, _ in number_structure_pairs])

    def sum(self,
            structure_expression_list: NestedList,
            relation_name: str) -> float:
        """
        Takes an expression list that evaluates to a list of structures and a relation name, and returns the sum
        of the values under that relation in those structures.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        if not number_structure_pairs:
            return 0.0
        return sum([value for value, _ in number_structure_pairs])

    def average(self,
                structure_expression_list: NestedList,
                relation_name: str) -> float:
        """
        Takes an expression list that evaluates to a list of structures and a relation name, and
        returns the average of the values under that relation in those structures.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        number_structure_pairs = [(structure[relation_name].numbers[0], structure) for
                                  structure in structure_list if
                                  structure[relation_name].numbers]
        if not number_structure_pairs:
            return 0.0
        return sum([value for value, _ in number_structure_pairs]) / len(number_structure_pairs)

    def mode(self,
             structure_expression_list: NestedList,
             relation_name: str) -> List[str]:
        """
        Takes an expression that evaluates to a list of structures, and a relation and returns the most
        frequent values (one or more) under that relation in those structures.
        """
        structure_list: List[Dict[str, Argument]] = self._handle_expression(structure_expression_list)
        if not structure_list:
            return []
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for structure in structure_list:
            argument_value = structure[relation_name].argument_string
            value_frequencies[argument_value] += 1
            frequency = value_frequencies[argument_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [argument_value]
            elif frequency == max_frequency:
                most_frequent_list.append(argument_value)
        return most_frequent_list

    def diff(self,
             first_value_expression_list: NestedList,
             second_value_expression_list: NestedList) -> float:
        """
        Takes two expressions that evaluate to floats and returns the difference between the values.
        """
        first_value = self._handle_expression(first_value_expression_list)
        if not isinstance(first_value, float):
            raise ExecutionError(f"Invalid expression for diff: {first_value_expression_list}")
        second_value = self._handle_expression(second_value_expression_list)
        if not isinstance(first_value, float):
            raise ExecutionError(f"Invalid expression for diff: {second_value_expression_list}")
        return first_value - second_value

    @staticmethod
    def date(year_string: str, month_string: str, day_string: str, quarter_string: str) -> Date:
        """
        Takes three numbers as strings, and returns a ``Date`` object whose year, month, and day are
        the three numbers in that order.
        """
        date_string = f"{year_string}-{month_string}-{day_string}-{quarter_string}"
        try:
            year = int(str(year_string))
            month = int(str(month_string))
            day = int(str(day_string))
            quarter = int(str(quarter_string))
            return Date(string=date_string, year=year, month=month, day=day, quarter=quarter)
        except ValueError:
            raise ExecutionError(f"Invalid date! Got: {date_string}")
