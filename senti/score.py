
import matplotlib; matplotlib.use('Agg')
from contextlib import closing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, auc, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from senti.utils import Tee

__all__ = ['write_score']


def write_score(name, gold_labels, pred_scores, classes, average_classes):
    gold_scores = LabelBinarizer().fit(classes).transform(gold_labels)
    pred_labels = classes[np.argmax(pred_scores, axis=1)]

    with closing(Tee('{}.txt'.format(name), 'w')):
        precision, recall, fscore, _ = precision_recall_fscore_support(gold_labels, pred_labels, labels=classes)
        for t in zip(classes, precision, recall, fscore):
            print('{}: P={:.2f}, R={:.2f}, F1={:.2f}'.format(*t))
        print('Accuracy: {:.4f}'.format(accuracy_score(gold_labels, pred_labels)))
        print('F1 average: {:.4f}'.format(np.mean(fscore[LabelEncoder().fit(classes).transform(average_classes)])))

    with PdfPages('{}.pdf'.format(name)) as pdf:
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(gold_scores[:, i], pred_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr['micro'], tpr['micro'], _ = roc_curve(gold_scores.ravel(), pred_scores.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        plt.figure()
        plt.plot(fpr['micro'], tpr['micro'], label='micro-average (area = {:.2f})'.format(roc_auc['micro']))
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], label='{0} (area = {1:.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        pdf.savefig()
