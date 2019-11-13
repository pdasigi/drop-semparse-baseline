# Semantic parsing baselines for DROP

Code for the semantic parsing baseline models sescribed in the following dataset paper by Dua et al.:

[**DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs**](https://www.semanticscholar.org/paper/DROP%3A-A-Reading-Comprehension-Benchmark-Requiring-Dua-Wang/dda6fb309f62e2557a071522354d8c2c897a2805)

Here are the instructions for using this code:

## 0. Install requirements

```
pip install -r ./requirements.txt
python -m spacy download en
```

## 1. Preprocessing

Transform passages into structured representations:

```
python scripts/preprocess_data.py data_file tables_output_path tagger
```

Where `tagger` indicates the underlying representation to use (one of `srl`, `dep`, or `oie`).

Alternatively, run preprocessing for all representations on the current train file, by running:

```
bash scripts/preprocess_data.sh data_file tables_output_path
```

## 2. Training

We will train a semantic parser from weak (denotation-only) supervision using Maximum Marginal Likelihood.
This requires a set of approximate logical forms (those that evaluate to the correct denotation, but do not necessarily have correct semantics) for each training instance.
You can generate those in the following way:

```
python scripts/search_for_logical_forms.py tables_output_path data_file logical_forms_output_path --use-agenda --output-separate-files --num-splits 10
```

Run `python scripts/search_for_logical_forms.py -h` to understand the arguments.

Finally, you can train a model using `allennlp` as follows:

```
allennlp train config.json -s training_output_path --include-package semparse
```

[Here](https://github.com/pdasigi/drop-semparse-baseline/blob/master/fixtures/model/experiment.json) is a sample configureation file.
