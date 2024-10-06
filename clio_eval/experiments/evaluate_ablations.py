import time
import argparse
import yaml
import json
import os
import subprocess
import shutil
import signal
import numpy as np
import csv
from tabulate import tabulate

import clio_eval.evaluate_helpers as eval_helpers
from clio_eval.utils import Ablations

def generate_raw_results(results, ablations, dec=3, print_table=True):
    folder = ablations.env["log_path"]
    header = ["Run", "IOU",
              "SAcc", "RAcc", "Sprec", "Rprec", "F1", "#-Objects"]
    fields = ["avg_iou", "strict_recall",
              "weak_recall", "strict_precision", "weak_precision", "f1", "num_objects"]
    # Maintain a latex table and a human readable table
    table = []
    table.append(header)

    for i in range(len(results)):
        trial_results = results[i]
        for run in trial_results:
            res = trial_results[run]
            row = ["{}-{}".format(i, run)] + res.to_list(fields, decimals=dec)
            table.append(row)

    output_txt = os.path.join(folder, "raw_results.txt")
    with open(output_txt, 'w') as f:
        f.write(tabulate(table))
    if print_table:
        print(tabulate(table))


def evaluate(ablations):
    results = []
    for trial in range(ablations.num_trials):
        trial_results = {}
        for dataset in ablations.experiments:
            for label in ablations.experiments[dataset].experiments:
                label = label[0]
                if "large" in label:
                    clip_model = "ViT-L/14"
                elif "open" in label:
                    clip_model = "ViT-H-14"
                else:
                    raise ValueError('Unable to determine clip mode.')
                task_yaml = ablations.experiments[dataset].task_yaml
                log_folder = "{}/{}/trial_{}/{}".format(
                    ablations.env["log_path"], dataset, trial, label)
                dsg_file = os.path.join(
                    log_folder, os.path.join("backend", "dsg.json"))
                if not os.path.exists(dsg_file):
                    print("Missing {}".format(dsg_file))
                    continue

                if "fine" in label:
                    fine_label = "{}_clio_primitives".format(dataset)
                    clio_label = "{}_clio".format(dataset)
                    if label.endswith("ps"):
                        fine_label += "_ps"
                        clio_label += "_ps"
                    trial_results[fine_label] = eval_helpers.results_from_files(
                        dsg_file, task_yaml, clip_model, False)
                    trial_results[clio_label] = eval_helpers.results_from_files(
                        dsg_file, task_yaml, clip_model, True, thresh=ablations.eval["semantic_threshold"])
                else:
                    print("Encountered unexpected label: {}".format(label))
        results.append(trial_results)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run ablations.")
    parser.add_argument('--configs', nargs='+', default=[])

    args = parser.parse_args()

    for config_yaml in args.configs:
        ablation_config = Ablations()
        ablation_config.parse(config_yaml)

        results = evaluate(ablation_config)
        generate_raw_results(results, ablation_config)

if __name__ == "__main__":
    main()