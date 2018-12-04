#!/bin/bash
# Usage:
#    preprocess.sh path/to/train/file path/to/output/folder
#
# Will create folders dep, oie, and srl in the output folder.

mkdir -p ${2}/dep ${2}/oie ${2}/srl
python scripts/preprocess_data.py $1 ${2}/dep/ dep
python scripts/preprocess_data.py $1 ${2}/oie/ oie
python scripts/preprocess_data.py $1 ${2}/srl/ srl
