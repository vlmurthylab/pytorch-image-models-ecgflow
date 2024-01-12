"""ROC etc.
"""
from collections import OrderedDict

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, \
    BinaryRecall, BinarySpecificity, BinaryROC, \
    MultilabelAccuracy, MultilabelAUROC, MultilabelRecall, \
    MultilabelSpecificity, MultilabelROC

BINARY_METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Prevalence',
                  'PPV', 'NPV', 'AUROC', 'F1-score', 'Thresh']


def get_binary_torchmetrics(prefix='', threshold=0.5, compute_on_cpu=True,
                            compute_with_cache=True):
    metrics = MetricCollection(
        dict(
            Accuracy=BinaryAccuracy(threshold=threshold,
                                    compute_on_cpu=compute_on_cpu,
                                    compute_with_cache=compute_with_cache),
            AUROC=BinaryAUROC(compute_on_cpu=compute_on_cpu,
                              compute_with_cache=compute_with_cache),
            ROC=BinaryROC(compute_on_cpu=compute_on_cpu,
                          compute_with_cache=compute_with_cache),
            Sensitivity=BinaryRecall(threshold=threshold,
                                     compute_on_cpu=compute_on_cpu,
                                     compute_with_cache=compute_with_cache),
            Specificity=BinarySpecificity(threshold=threshold,
                                          compute_on_cpu=compute_on_cpu,
                                          compute_with_cache=compute_with_cache)
        ),
        prefix=prefix)
    return metrics


def get_multilabel_torchmetrics(num_labels, prefix='', threshold=0.5,
                                compute_on_cpu=True, compute_with_cache=True):
    metrics = MetricCollection(
        dict(
            Accuracy=MultilabelAccuracy(num_labels, threshold=threshold,
                                    compute_on_cpu=compute_on_cpu,
                                    compute_with_cache=compute_with_cache),
            AUROC=MultilabelAUROC(num_labels, compute_on_cpu=compute_on_cpu,
                              compute_with_cache=compute_with_cache),
            ROC=MultilabelROC(num_labels, compute_on_cpu=compute_on_cpu,
                          compute_with_cache=compute_with_cache),
            Sensitivity=MultilabelRecall(num_labels, threshold=threshold,
                                     compute_on_cpu=compute_on_cpu,
                                     compute_with_cache=compute_with_cache),
            Specificity=MultilabelSpecificity(num_labels, threshold=threshold,
                                          compute_on_cpu=compute_on_cpu,
                                          compute_with_cache=compute_with_cache)
        ),
        prefix=prefix)
    return metrics


def get_binary_torchmetrics_odict(prefix='', threshold=0.5, compute_on_cpu=True,
                                  compute_with_cache=False):
    metrics = OrderedDict(
            Accuracy=BinaryAccuracy(threshold=threshold,
                                    compute_on_cpu=compute_on_cpu),
            AUROC=BinaryAUROC(compute_on_cpu=compute_on_cpu),
            ROC=BinaryROC(compute_on_cpu=compute_on_cpu),
            Sensitivity=BinaryRecall(threshold=threshold,
                                     compute_on_cpu=compute_on_cpu),
            Specificity=BinarySpecificity(threshold=threshold,
                                          compute_on_cpu=compute_on_cpu)
        )
    return metrics
