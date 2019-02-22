#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.
# pylint: disable=all

import argparse
import os
import json
import random
import tempfile
import subprocess
import sys

# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.params import Params


def main(args: argparse.Namespace):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"allennlp/allennlp:{commit}"
    overrides = ""


    # If the git repository is dirty, add a random hash.
    result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        image += "-" + dirty_hash

    if args.blueprint:
        blueprint = args.blueprint
        print(f"Using the specified blueprint: {blueprint}")
    else:
        print(f"Building the Docker image ({image})...")
        subprocess.run(f'docker build -t {image} .', shell=True, check=True)

        print(f"Create a Beaker blueprint...")
        blueprint = subprocess.check_output(f'beaker blueprint create --quiet {image}', shell=True, universal_newlines=True).strip()
        print(f"  Blueprint created: {blueprint}")


    dataset_mounts = []
    if args.evaluation_script is None:
        assert args.experiment_type == "preprocess", "Evaluation script is required for training and search!"
    else:
        evaluator_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {args.evaluation_script}', shell=True, universal_newlines=True).strip()
        dataset_mounts.append({"datasetId": evaluator_dataset_id,
                               "containerPath": "/evaluate.py"})


    if args.data_file_path is None:
        assert args.experiment_type == "training", "Path to data files required for search and preprocessing!"
    else:
        data_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {args.data_file_path}', shell=True, universal_newlines=True).strip()
        dataset_mounts.append({"datasetId": data_dataset_id,
                               "containerPath": "/data_files/"})

    for source in args.source:
        datasetId, containerPath = source.split(":")
        dataset_mounts.append({
            "datasetId": datasetId,
            "containerPath": containerPath
        })

    env = {}

    if args.experiment_type == "training":
        # Reads params and sets environment.
        params = Params.from_file(args.param_file, overrides)
        flat_params = params.as_flat_dict()
        for k, v in flat_params.items():
            k = str(k).replace('.', '_')
            env[k] = str(v)
        config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {args.param_file}', shell=True, universal_newlines=True).strip()
        dataset_mounts.append({"datasetId": config_dataset_id,
                               "containerPath": "/config.json"})
        command = [
                "python",
                "-m",
                "allennlp.run",
                "train",
                "/config.json",
                "-s",
                "/output",
                "--file-friendly-logging",
                "--include-package",
                "semparse"
            ]

    elif args.experiment_type == "search":
        # TODO (pradeep): Expose more options below (e.g.: using agenda, num_splits, path_length etc.)
        command = [
                "python",
                "scripts/search_for_logical_forms.py",
                "/tables",
                "/data_files",
                "/output/logical_forms",
                "--max-path-length",
                "10",
                "--output-separate-files",
                "--num-splits",
                "10"
            ]
        if args.use_embedding:
            # Hardcoding Matt's Glove dataset on Beaker. Change this if you need to.
            dataset_mounts.append({"datasetId": "ds_0gnx4e9o3ap1",
                                   "containerPath": "/glove_vectors/"})
            command.extend(["--embedding-file",
                            "/glove_vectors/glove.6B.50d.txt.gz",
                            "--distance-threshold",
                            args.distance_threshold])
    elif args.experiment_type == "preprocess":
        command = [
                "python",
                "scripts/preprocess_data.py",
                "/data_files",
                "/output/",
                args.preprocessing_tagger_type,
            ]
        if args.include_coref_while_preprocessing:
            command.append('--include-coref')

    else:
        raise RuntimeError(f"Unknown experiment type: {args.experiemnt_type}")

    for var in args.env:
        key, value = var.split("=")
        env[key] = value

    requirements = {}
    if args.cpu:
        requirements["cpu"] = float(args.cpu)
    if args.memory:
        requirements["memory"] = args.memory
    if args.gpu_count:
        requirements["gpuCount"] = int(args.gpu_count)
    config_spec = {
        "description": args.desc,
        "blueprint": blueprint,
        "resultPath": "/output",
        "args": command,
        "datasetMounts": dataset_mounts,
        "requirements": requirements,
        "env": env
    }
    config_task = {"spec": config_spec, "name": args.experiment_type}

    config = {
        "tasks": [config_task]
    }

    output_path = args.spec_output_path if args.spec_output_path else tempfile.mkstemp(".yaml",
            "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--file", output_path]
    if args.name:
        experiment_command.append("--name")
        experiment_command.append(args.name.replace(" ", "-"))

    if args.dry_run:
        print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
        print(f"    " + " ".join(experiment_command))
    else:
        print(f"Running the experiment:")
        print(f"    " + " ".join(experiment_command))
        subprocess.run(experiment_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-script', dest='evaluation_script', type=str,
                        help='Path to the official evaluation script, required for training and search.')
    parser.add_argument('--experiment-type', dest="experiment_type", type=str,
                        choices=['training', 'search', 'preprocess'], default='training',
                        help='Are you training a model, searching for logical forms, or preprocessing data on beaker?')
    parser.add_argument('--param-file', dest="param_file", type=str,
                        help='The model configuration file, required only for training')
    parser.add_argument('--data-file-path', dest="data_file_path", type=str,
                        help='Path to the directory with the original dataset files; required only for search or preprocessing')
    parser.add_argument('--preprocessing-tagger-type', dest="preprocessing_tagger_type", type=str,
                        choices=['dep', 'oie', 'srl'], default='srl',
                        help='Type of tagger to use for extracting predicate argument structures')
    parser.add_argument('--include-coref-while-preprocessing', dest='include_coref_while_preprocessing',
                        action='store_true', help='Should we include coref information while performing IE on paragraphs?')
    parser.add_argument('--use-embedding-for-search', dest="use_embedding", action='store_true',
                        help='''Should we use an embedding file to get better context representations during search?
                                Glove's 50d vectors will be used.''')
    parser.add_argument("--distance-threshold-for-similarity", dest="distance_threshold", type=float, default=0.3,
                        help="Value to use as threshold for measuring similarity with embedding (default 0.3).")
    parser.add_argument('--name', type=str, help='A name for the experiment.')
    parser.add_argument('--spec_output_path', type=str, help='The destination to write the experiment spec.')
    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    parser.add_argument('--blueprint', type=str, help='The Blueprint to use (if unspecified one will be built)')
    parser.add_argument('--desc', type=str, help='A description for the experiment.')
    parser.add_argument('--env', action='append', default=[], help='Set environment variables (e.g. NAME=value or NAME)')
    parser.add_argument('--source', action='append', default=[], help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--cpu', help='CPUs to reserve for this experiment (e.g., 0.5)')
    parser.add_argument('--gpu-count', default=1, help='GPUs to use for this experiment (e.g., 1 (default))')
    parser.add_argument('--memory', help='Memory to reserve for this experiment (e.g., 1GB)')

    args = parser.parse_args()

    main(args)
