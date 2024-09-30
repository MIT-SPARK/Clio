import numpy as np
import clio_eval.utils as eval_utils
import builtins


class StatValue:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def from_list(cls, l):
        values = np.array(l)
        return cls(np.mean(values), np.std(values))

    def __gt__(self, other):
        if type(other) == StatValue:
            return self.mean > other.mean
        # assume if not StatValue then some scalar
        return self.mean > other

    def __round__(self, precision):
        if precision <= 0:
            self.mean = int(self.mean)
            self.std = int(self.std)
        else:
            self.mean = round(self.mean, precision)
            self.std = round(self.std, precision)
        return self

    def __str__(self):
        return str(self.mean) + r" $\pm$ " + str(self.std)


def bold_optimal_value(table, compr):
    def get_value(val):
        if type(val) == StatValue:
            return val.mean
        return val

    # table is list of lists
    # compr list of signs telling us how to compare values
    for i in range(len(compr)):
        if compr[i] == 0:
            continue

        values = [row[i] for row in table]
        if compr[i] > 0:
            opt_val = max(values)
        elif compr[i] < 0:
            opt_val = min(values)
        opt_idx = values.index(opt_val)
        table[opt_idx][i] = r"\textbf{" + str(table[opt_idx][i]) + r"}"


def multi_trial_results(results_list):
    if len(results_list) == 1:
        return results_list[0]
    objects = StatValue.from_list([r.num_objects for r in results_list])
    w_recall = StatValue.from_list([r.weak_recall for r in results_list])
    s_recall = StatValue.from_list([r.strict_recall for r in results_list])
    avg_iou = StatValue.from_list([r.avg_iou for r in results_list])
    w_prec = StatValue.from_list([r.weak_precision for r in results_list])
    s_prec = StatValue.from_list([r.strict_precision for r in results_list])
    f1 = StatValue.from_list([r.f1 for r in results_list])
    tpf = StatValue.from_list([r.tpf for r in results_list])
    return eval_utils.Results(objects, w_recall, s_recall, avg_iou, w_prec, s_prec, f1, tpf)
