import numpy as np
import matplotlib
import matplotlib.pylab as plt
import pandas as pd
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             auc,
                             average_precision_score,
                             confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_recall_curve)

plt.rcParams['figure.figsize'] = (16, 8)

#------- History Visualizations -----------

def plot_f1_score(history_obj):
    plt.plot(history_obj.history['f1'])
    plt.plot(history_obj.history['val_f1'])
    plt.title('model f1')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()

# summarize history for f1y
def plot_loss(history_obj):
    # summarize history for loss
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper right')
    plt.show()


def plot_accuracy(history_obj):
    # summarize history for accurracy
    plt.plot(history_obj.history['accuracy'])
    plt.plot(history_obj.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper right')
    plt.show()


def plot_history_metrics(history_obj):
    plot_f1_score(history_obj)

    plot_accuracy(history_obj)

#------- Prediction's Visualizations -----------

def create_clf_report(y_true, y_pred, classes):
    """
    This function calculates several metrics about a classifier and creates a mini report.
    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                             index=classes,
                             columns=['pre_{}'.format(c) for c in classes])

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion


def plot_roc_curve(y_true, y_pred_scores, nclasses, pos_label=1):
    """
    :param y_true:
    :param y_pred_scores:
    :param pos_label:
    :return:
    """
    for cls_index, cls in enumerate(nclasses):
        fpr, tpr, _ = roc_curve(y_true[:, cls_index], y_pred_scores[:, cls_index], pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 1
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for class "%s"' % cls.capitalize())
        plt.legend(loc="lower right")
        plt.show()


def plot_precision_recall_curve(y_true, y_pred_scores, nclasses, pos_label=1):
    """
    :param y_true:
    :param y_pred_scores:
    :param pos_label:
    :return:
    """
    for cls_index, cls in enumerate(nclasses):

        average_precision = average_precision_score(y_true[:, cls_index],
                                                    y_pred_scores[:, cls_index])
        precision, recall, _ = precision_recall_curve(y_true[:, cls_index],
                                                      y_pred_scores[:, cls_index],
                                                      pos_label=pos_label)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve for class "{0}": AP={1:0.2f}'.format(cls.capitalize(),
                                                                             average_precision))


def plot_prediction_metrics(y_true, y_pred, nclasses):
    """
    :param y_true:
    :param y_pred:
    :return:
    """

    import ipdb
    ipdb.set_trace()
    y_true_processed = np.array([np.argmax(val) + 1 for val in y_true])
    y_pred_processed = np.array([np.argmax(val) + 1 for val in y_pred])

    create_clf_report(y_true_processed, y_pred_processed, nclasses)

    plot_roc_curve(y_true, y_pred, nclasses, 1)

    plot_prediction_metrics(y_true_processed, y_pred_processed)
