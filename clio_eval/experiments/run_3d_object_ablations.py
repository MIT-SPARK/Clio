import argparse
import yaml
import os
import csv
import json
import numpy as np
from tabulate import tabulate
import subprocess

import clio_batch.object_cluster as obj_cluster
import clio_eval.evaluate_helpers as eval_helpers
import clio_eval.utils as eval_utils
from clio_eval.evaluate_helpers import get_dsg_version


def parse_experiment_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)

def write_csv(table, filename):
    # Dominic's write csv to interface with latex converter in paper repo
    # Define the order of columns
    column_order = ["IOU", "S. Rec", "W. Rec",
                    "S. Prec", "W. Prec", "F1", "#-Objs", "Run"]

    # Reorder the columns in each row of the table
    reordered_table = []
    for row in table:
        reordered_row = [row[table[0].index(col)] for col in column_order]
        reordered_table.append(reordered_row)

    # Write the reordered table to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(reordered_table)


def print_and_save_results(results, folder, dec=3):
    table = []
    header = ["Run", "#-Objs", "IOU",
              "S. Rec", "W. Rec", "S. Prec", "W. Prec", "F1"]

    table.append(header)

    for run in results:
        res = results[run]
        row = [run, res.num_objects, round(res.avg_iou, dec), round(res.strict_recall, dec), round(
            res.weak_recall, dec), round(res.strict_precision, dec), round(res.weak_precision, dec), round(res.f1, dec)]
        table.append(row)

    write_csv(table, 'results.csv')
    print(tabulate(table))
    output_file = os.path.join(folder, "results.txt")
    with open(output_file, 'w') as f:
        f.write(tabulate(table))


def run(ablation_dict):
    for dataset in ablation_dict["datasets"]:
        clip_model = dataset["clip_model"]
        for experiment in dataset["experiments"]:
            output_folder = "{}/{}/{}".format(
                ablation_dict["log_path"], dataset["name"], experiment["name"])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            elif os.path.exists(os.path.join(output_folder, "dsg.json")):
                print(
                    "{} - {} exists. Skipping...".format(dataset["name"], experiment["name"]))
                continue
            experiments_path = os.path.abspath(os.path.dirname(__file__))
            cluster_config_dir = os.path.join(
                experiments_path, "configs/cluster")
            cluster_config_path = os.path.join(
                cluster_config_dir, experiment["cluster_config"])
            clustered_dsg = obj_cluster.cluster_3d(
                dataset["fine_dsg"], dataset["task_yaml"], output_folder, cluster_config_path, clip_model,
                experiment["prune_threshold"], experiment["partition"], experiment["bbox_dilation"],
                recompute_edges=experiment["recompute_edges"])
            output_dsg = output_folder + "/dsg.json"
            clustered_dsg.save(output_dsg)

            # Keep output DSG version the same as input version.
            version = get_dsg_version(dataset["fine_dsg"])
            print(version)
            with open(output_dsg) as file:
                d = json.load(file)
            version = [int(i) for i in version]
            d['SPARK_ORIGIN_header'] = {'version':{'major':version[0], 'minor':version[1], 'patch':version[2]}, 'project_name':'main'}

            with open(output_dsg, 'w') as file:
                json.dump(d, file) 

def evaluate(ablation_dict):
    results = {}
    for dataset in ablation_dict["datasets"]:
        clip_model = dataset["clip_model"]
        task_yaml = dataset["task_yaml"]
        fine_label = "{}_fine".format(dataset["name"])
        results[fine_label] = eval_helpers.results_from_files(
            dataset["fine_dsg"], task_yaml, clip_model, False)
        khronos_label = "{}_khronos".format(dataset["name"])
        results[khronos_label] = eval_helpers.results_from_files(
            dataset["khronos_dsg"], task_yaml, clip_model, False)
        medium_thres_label = "{}_khronos_thres".format(dataset["name"])
        # assumes threshold for Khronos is the same as the first thresh in experiment yaml
        thresh = dataset['experiments'][0]["prune_threshold"]
        results[medium_thres_label] = eval_helpers.results_from_files(
            dataset["khronos_dsg"], task_yaml, clip_model, False, thresh=thresh)
        for experiment in dataset["experiments"]:
            output_folder = "{}/{}/{}".format(
                ablation_dict["log_path"], dataset["name"], experiment["name"])
            dsg_file = os.path.join(output_folder, "dsg.json")
            if not os.path.exists(dsg_file):
                print("Missing {}".format(dsg_file))
                continue

            run_label = "{}_{}".format(dataset["name"], experiment["name"])
            # 3D clustering currently store clusters in places layer
            results[run_label] = eval_helpers.results_from_files(
                dsg_file, task_yaml, clip_model, True)

    print_and_save_results(results, ablation_dict["log_path"])


def main():
    parser = argparse.ArgumentParser(description="Run ablations.")
    parser.add_argument("config", type=str, default=None)

    args = parser.parse_args()
    ablation_dict = parse_experiment_yaml(args.config)

    print("Running clustering...")
    run(ablation_dict)

    print("Evaluating...")
    evaluate(ablation_dict)


if __name__ == "__main__":
    main()