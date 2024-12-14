# Script for MultiSpanQA evaluation
import os
import re
import json
import string
import difflib
import warnings
import numpy as np

import utils


def compute_scores(golds, preds, eval_type='em'):
    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for k in list(golds.keys()):
        gold = golds[k]
        pred = preds[k]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type == 'em':
            if len(gold) == 0 and len(pred) == 0:
                nb_correct += 1
            else:
                nb_correct += len(gold.intersection(pred))
        else:
            p_score, r_score = count_overlap(gold, pred)
            nb_correct_p += p_score
            nb_correct_r += r_score

    if eval_type == 'em':
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_gold if nb_gold > 0 else 0
    else:
        p = nb_correct_p / nb_pred if nb_pred > 0 else 0
        r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f


def count_overlap(gold, pred):
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, s1 in enumerate(gold):
        for j, s2 in enumerate(pred):
            s = difflib.SequenceMatcher(None, s1, s2)
            _, _, longest = s.find_longest_match(0, len(s1), 0, len(s2))
            p_scores[i][j] = longest/len(s2) if longest > 0 else 0
            r_scores[i][j] = longest/len(s1) if longest > 0 else 0

    p_score = sum(np.max(p_scores, axis=0))
    r_score = sum(np.max(r_scores, axis=1))

    return p_score, r_score


def multi_span_evaluate(preds, golds):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()
    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: utils.clean_italian_span(x).lower(), v))
    for k, v in preds.items():
        preds[k] = set(map(lambda x: utils.clean_italian_span(x).lower(), v))

    # Evaluate
    em_p, em_r, em_f = compute_scores(golds, preds, eval_type='em')
    overlap_p, overlap_r, overlap_f = compute_scores(golds, preds, eval_type='overlap')
    result = {'em_precision': 100*em_p,
              'em_recall': 100*em_r,
              'em_f1': 100*em_f,
              'overlap_precision': 100*overlap_p,
              'overlap_recall': 100*overlap_r,
              'overlap_f1': 100*overlap_f}
    return result, golds, preds





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default="", type=str)
    parser.add_argument('--gold_file', default="", type=str)
    args = parser.parse_args()
    result = multi_span_evaluate_from_file(args.pred_file, args.gold_file)
    for k, v in result.items():
        print(f"{k}: {v}")
