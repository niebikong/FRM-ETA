import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import sklearn.metrics as sk


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def print_measures(mylog, auroc, aupr, fpr):
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))
    mylog.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


def show_performance(pos, neg, method_name='Ours', recall_level=0.95):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    samples. This is the positive class.
    :param neg: 0's class, class to not detect, inliers, or correctly predicted
    samples. This is the negative class.
    :param method_name: name of the method
    :return: None
    '''
    auroc, aupr, fpr = get_measures(pos, neg, recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    print('')

    return fpr, auroc, aupr


def cal_metric(known, novel):
    tp, fp, tn, fn = 0, 0, 0, 0
    all_known = np.array(known)
    all_novel = np.array(novel)
    
    known_sorted = np.sort(all_known)
    novel_sorted = np.sort(all_novel)
    
    num_k = all_known.shape[0]
    num_n = all_novel.shape[0]
    
    threshold = known_sorted[round(0.05 * num_k)]
    
    y_true = np.concatenate((np.zeros(num_k), np.ones(num_n)))
    y_score = np.concatenate((all_known, all_novel))
    
    fpr_temp, tpr_temp, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    
    fpr95 = fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95)
    
    return {
        'FPR95': fpr95,
        'AUROC': auroc,
        'AUPR': aupr
    }


def print_all_results(results, ood_dataset_names, method_name):
    print(f'\n{method_name} Results:')
    print('=' * 50)
    for i, ood_name in enumerate(ood_dataset_names):
        result = results[i]
        print(f'{ood_name}:')
        print(f'  FPR95: {result["FPR95"]:.4f}')
        print(f'  AUROC: {result["AUROC"]:.4f}')
        print(f'  AUPR:  {result["AUPR"]:.4f}')
        print('-' * 30)
