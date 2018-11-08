"""
Defines a language for performing discrete reasoning over paragraphs.
"""


from allennlp.semparse.type_declarations.type_declaration import (NamedBasicType, ComplexType,
                                                                  NameMapper)


# Basic types
PAS_TYPE = NamedBasicType("PAS")  # Predicate Argument Structure
RELATION_TYPE = NamedBasicType("RELATION")

NUMBER_TYPE = NamedBasicType("NUMBER")
DATE_TYPE = NamedBasicType("DATE")
STRING_TYPE = NamedBasicType("STRING")

BASIC_TYPES = {PAS_TYPE, RELATION_TYPE, NUMBER_TYPE, DATE_TYPE, STRING_TYPE}
STARTING_TYPES = {NUMBER_TYPE, DATE_TYPE, STRING_TYPE}

# Complex types
# Type for selecting the value in a column in a set of rows. "select", "mode", and "extract_entity" functions.
SELECT_TYPE = ComplexType(PAS_TYPE, ComplexType(RELATION_TYPE, STRING_TYPE))

# Type for filtering structures given a relation. "argmax", "argmin"
PAS_FILTER_WITH_RELATION = ComplexType(PAS_TYPE, ComplexType(RELATION_TYPE, PAS_TYPE))

# "filter_number_greater", "filter_number_equals" etc.
PAS_FILTER_WITH_RELATION_AND_NUMBER = ComplexType(PAS_TYPE,
                                                  ComplexType(RELATION_TYPE,
                                                              ComplexType(NUMBER_TYPE, PAS_TYPE)))

# "filter_date_greater", "filter_date_equals" etc.
PAS_FILTER_WITH_RELATION_AND_DATE = ComplexType(PAS_TYPE,
                                                ComplexType(RELATION_TYPE,
                                                            ComplexType(DATE_TYPE, PAS_TYPE)))

# "filter_in" and "filter_not_in"
PAS_FILTER_WITH_RELATION_AND_STRING = ComplexType(PAS_TYPE,
                                                  ComplexType(RELATION_TYPE,
                                                              ComplexType(STRING_TYPE, PAS_TYPE)))

PAS_FILTER = ComplexType(PAS_TYPE, PAS_TYPE)  # first, last, previous, next etc.

PAS_COUNT_TYPE = ComplexType(PAS_TYPE, NUMBER_TYPE)

# Numerical operations on numbers within the given relation. "max", "min", "sum", "average",
# "count_entities", "extract_number".
PAS_NUM_OP = ComplexType(PAS_TYPE, ComplexType(RELATION_TYPE, NUMBER_TYPE))

# Numerical difference (diff)
NUM_DIFF_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, NUMBER_TYPE))

# Date function takes four numbers and makes a date
DATE_FUNCTION_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE,
                                                          ComplexType(NUMBER_TYPE,
                                                                      ComplexType(NUMBER_TYPE,
                                                                                  DATE_TYPE))))

generic_name_mapper = NameMapper()  # pylint: disable=invalid-name

generic_name_mapper.map_name_with_signature("all_structures", PAS_TYPE)

# <p,<r,s>>
generic_name_mapper.map_name_with_signature("select", SELECT_TYPE)
generic_name_mapper.map_name_with_signature("mode", SELECT_TYPE)
generic_name_mapper.map_name_with_signature("extract_entity", SELECT_TYPE)

# <p,<r,p>>
generic_name_mapper.map_name_with_signature("argmax", PAS_FILTER_WITH_RELATION)
generic_name_mapper.map_name_with_signature("argmin", PAS_FILTER_WITH_RELATION)

# <p,<r,<n,p>>>
generic_name_mapper.map_name_with_signature("filter_number_greater", PAS_FILTER_WITH_RELATION_AND_NUMBER)
generic_name_mapper.map_name_with_signature("filter_number_greater_equals", PAS_FILTER_WITH_RELATION_AND_NUMBER)
generic_name_mapper.map_name_with_signature("filter_number_lesser", PAS_FILTER_WITH_RELATION_AND_NUMBER)
generic_name_mapper.map_name_with_signature("filter_number_lesser_equals", PAS_FILTER_WITH_RELATION_AND_NUMBER)
generic_name_mapper.map_name_with_signature("filter_number_equals", PAS_FILTER_WITH_RELATION_AND_NUMBER)
generic_name_mapper.map_name_with_signature("filter_number_not_equals", PAS_FILTER_WITH_RELATION_AND_NUMBER)

# <p,<r,<d,p>>>
generic_name_mapper.map_name_with_signature("filter_date_greater", PAS_FILTER_WITH_RELATION_AND_DATE)
generic_name_mapper.map_name_with_signature("filter_date_greater_equals", PAS_FILTER_WITH_RELATION_AND_DATE)
generic_name_mapper.map_name_with_signature("filter_date_lesser", PAS_FILTER_WITH_RELATION_AND_DATE)
generic_name_mapper.map_name_with_signature("filter_date_lesser_equals", PAS_FILTER_WITH_RELATION_AND_DATE)
generic_name_mapper.map_name_with_signature("filter_date_equals", PAS_FILTER_WITH_RELATION_AND_DATE)
generic_name_mapper.map_name_with_signature("filter_date_not_equals", PAS_FILTER_WITH_RELATION_AND_DATE)

# <p,<r,<s,p>>>
generic_name_mapper.map_name_with_signature("filter_in", PAS_FILTER_WITH_RELATION_AND_STRING)
generic_name_mapper.map_name_with_signature("filter_not_in", PAS_FILTER_WITH_RELATION_AND_STRING)

# <p,p>
generic_name_mapper.map_name_with_signature("first", PAS_FILTER)
generic_name_mapper.map_name_with_signature("last", PAS_FILTER)
generic_name_mapper.map_name_with_signature("previous", PAS_FILTER)
generic_name_mapper.map_name_with_signature("next", PAS_FILTER)

# <p,n>
generic_name_mapper.map_name_with_signature("count_structures", PAS_COUNT_TYPE)

# <p,<r,n>>
generic_name_mapper.map_name_with_signature("count_entities", PAS_NUM_OP)
generic_name_mapper.map_name_with_signature("extract_number", PAS_NUM_OP)
generic_name_mapper.map_name_with_signature("max", PAS_NUM_OP)
generic_name_mapper.map_name_with_signature("min", PAS_NUM_OP)
generic_name_mapper.map_name_with_signature("average", PAS_NUM_OP)
generic_name_mapper.map_name_with_signature("sum", PAS_NUM_OP)

# <n,<n,n>>
generic_name_mapper.map_name_with_signature("diff", NUM_DIFF_TYPE)

# <n,<n,<n,<n,d>>>>
generic_name_mapper.map_name_with_signature("date", DATE_FUNCTION_TYPE)


COMMON_NAME_MAPPING = generic_name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = generic_name_mapper.common_type_signature
