# Semantic parsing baselines

0. Install requirements

```
pip install -r ./requirements.txt
python -m spacy download en
```

1. Preprocessing

```
python scripts/preprocess_data.py data_file output_path tagger
```

Where `tagger` indicates the underlying representation to use (one of `srl`, `dep`, or `oie`).

Alternatively, run preprocessing for all representations on the current train file, by running:

```
bash scripts/preprocess_data.sh path/to/train/file path/to/output/folder
```

2. Training

3. Testing
