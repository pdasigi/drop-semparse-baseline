from typing import Dict
import gzip

import numpy


def read_pretrained_embedding(embedding_file_path: str) -> Dict[str, numpy.ndarray]:
    embedding: Dict[str, numpy.ndarray] = {}
    with gzip.open(embedding_file_path, "rt") as embedding_file:
        for line in embedding_file:
            line_parts = line.split(' ')
            token = line_parts[0]
            token_embedding = numpy.asarray(line_parts[1:], dtype='float32')
            embedding[token] = token_embedding
    print(f"Read pretrained embedding file with {len(embedding)} tokens.")
    return embedding
