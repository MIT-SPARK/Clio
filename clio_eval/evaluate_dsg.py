import clio_eval.evaluate_helpers as eval_helpers
import clio_eval.utils as eval_utils
import spark_dsg as sdsg
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DSG. ")
    parser.add_argument('dsg_file', type=str, default=None)
    parser.add_argument('gt_yaml', type=str, default=None)
    parser.add_argument('-t', '--thres', type=float, default=0.0)
    parser.add_argument('--clip_model', type=str,
                        default="ViT-L/14", help="ViT-L/14, ViT-B/32")
    parser.add_argument('--clustered',
                        default=False, action="store_true")
    parser.add_argument('-v', '--visualize',
                        default=False, action="store_true")
    

    args = parser.parse_args()

    results = eval_helpers.results_from_files(
        args.dsg_file, args.gt_yaml, args.clip_model, args.clustered, args.visualize, args.thres)

    print(results)


if __name__ == "__main__":
    main()
