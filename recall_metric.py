# coding=utf-8
""" DIST metric """

import datasets
import re


_CITATION = """citation
"""

_DESCRIPTION = """\
Recall of ReDial
"""

_KWARGS_DESCRIPTION = """
args and examples
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Recall(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(
                        datasets.Sequence(datasets.Value("string", id="token"), id="sequence"), id="predictions"
                    ),
                    "references": datasets.Sequence(datasets.Value("string", id="token"), id="sequence"),
                }
            ),
        )

    def _compute_recall(self, labels, outs, partially_matching=False):
        if len(labels) == 1 and not labels[0]:
            return 0, 0, 0
        recall_count = [0, 0, 0]
        list_num = [1, 10, 50]
        for num_idx, num in enumerate(list_num):
            if len(outs) >= num:
                for label in labels:
                    for out in outs[:num]:
                        # partially matching
                        if partially_matching:
                            label_word = label.split(" ")
                            matching = False
                            for word in label_word:
                                if re.sub(r'[^(\w|\s)]', '', word).lower() in [elem.lower() for elem in out]:
                                    recall_count[num_idx] += 1
                                    matching = True
                                    break
                            if matching:
                                break
                        # fully matching
                        elif re.sub(r'[^(\w|\s)]', '', label).lower() in " ".join(out).lower():
                            recall_count[num_idx] += 1
                            break
        recall_1 = 0 if recall_count[0] == 0 else recall_count[0]/len(labels)
        recall_10 = 0 if recall_count[1] == 0 else recall_count[1]/len(labels)
        recall_50 = 0 if recall_count[2] == 0 else recall_count[2]/len(labels)
        return recall_1, recall_10, recall_50


    def _compute(self, predictions, references, partially_matching=False):
        list_recall_1 = 0
        list_recall_10 = 0
        list_recall_50 = 0
        n_count = 0
        for ref, pred in zip(references, predictions):
            if len(ref) == 1 and not ref[0]:
                continue
            recall_1, recall_10, recall_50 = self._compute_recall(ref, pred, partially_matching=partially_matching)
            if recall_1 != 0:
                print("pred")
                print(pred)
                print("ref")
                print(ref)
            n_count += 1
            list_recall_1 += recall_1
            list_recall_10 += recall_10
            list_recall_50 += recall_50
        res = {}
        res["recall_1"] = list_recall_1/n_count if n_count != 0 else 0
        res["recall_10"] = list_recall_10/n_count if n_count != 0 else 0
        res["recall_50"] = list_recall_50/n_count if n_count != 0 else 0
        return res