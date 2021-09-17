# coding=utf-8
""" DIST metric """

import datasets


_CITATION = """citation
"""

_DESCRIPTION = """\
DIST
"""

_KWARGS_DESCRIPTION = """
args and examples
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Dist(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="token"), id="sequence"),
                    "references": datasets.Sequence(
                        datasets.Sequence(datasets.Value("string", id="token"), id="sequence"), id="references"
                    ),
                }
            ),
        )

    def _compute(self, predictions, references, sentence_dist = True):
        if not sentence_dist:
            unigram_score = []
            bigram_score = []
            trigram_score = []
            quagram_score = []
            for sen in predictions:
                unigram_set = set()
                bigram_set = set()
                trigram_set=set()
                quagram_set=set()
                for word in sen:
                    unigram_set.add(word)
                if len(unigram_set) == 0:
                    unigram_score.append(0)
                else:
                    unigram_score.append(len(unigram_set)/len(sen))

                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_set.add(bg)
                if len(bigram_set) == 0:
                    bigram_score.append(0)
                else:
                    bigram_score.append(len(bigram_set)/len(sen))

                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_set.add(trg)
                if len(trigram_set) == 0:
                    trigram_score.append(0)
                else:
                    trigram_score.append(len(trigram_set)/len(sen))

                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_set.add(quag)
                if len(quagram_set) == 0:
                    quagram_score.append(0)
                else:
                    quagram_score.append(len(quagram_set)/len(sen))
                
            dis1 = sum(unigram_score) / len(predictions)
            dis2 = sum(bigram_score) / len(predictions)
            dis3 = sum(trigram_score) / len(predictions)
            dis4 = sum(quagram_score) / len(predictions)
            res = {}
            res["dis1"] = dis1
            res["dis2"] = dis2
            res["dis3"] = dis3
            res["dis4"] = dis4
            return res
        else:
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in predictions:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(predictions)
            dis2 = len(bigram_set) / len(predictions)
            dis3 = len(trigram_set) / len(predictions)
            dis4 = len(quagram_set) / len(predictions)
            res = {}
            res["dis1"] = dis1
            res["dis2"] = dis2
            res["dis3"] = dis3
            res["dis4"] = dis4
            return res