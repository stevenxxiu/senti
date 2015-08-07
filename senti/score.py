
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, precision_recall_fscore_support, roc_curve


def write_score(name, gold_labels, pred_scores, classes, average_classes):
    gold_scores = preprocessing.label_binarize(gold_labels, classes)
    pred_labels = list(classes[i] for i in np.argmax(pred_scores, axis=1))

    precision, recall, fscore, _ = precision_recall_fscore_support(pred_labels, gold_labels)
    for t in zip(classes, precision, recall, fscore):
        print('{}: P={:.2f}, R={:.2f}, F1={:.2f}'.format(*t))
    print('Accuracy: ', accuracy_score(pred_labels, gold_labels))
    print('F1 average: ', np.mean(list(fscore[classes.index(c)] for c in average_classes)))

    with PdfPages(name) as pdf:
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
