from .eclogging import load_logger
import itertools
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re

logger = load_logger()

###
def pr_auc_score(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    
    return pr_auc


### 
def pr_curve(y_test, y_score):
    """
    plot precision-recall curve and return AUC of it.
    
    Parameters
    ----------
    y_test = [n_samples] shape
    y_score = [n_samples] shape
    """

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot([1, 0], [0, 1], color=(0.85, 0.85, 0.85), lw=2, linestyle='--')
    
    plt.step(recall, precision, where='post', label='PR curve (area = %0.2f)' % pr_auc, color = (0.22, 0.42, 0.69)
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
    'Precision-Recall Curve'
    )
    plt.legend(loc='lower right', fontsize='large')

    return pr_auc


### 
def roc_plot(y_test, y_score, filename=None):
    ##Computing false and true positive rates
    fpr, tpr,_=roc_curve(y_test, y_score, drop_intermediate=False)

    plt.figure()
    ##Adding the ROC
    plt.plot(fpr, tpr, color='darkorange',
     lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_score))
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ##Title and label
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Receiver Operating Chracteristic curve')
    plt.legend(loc='lower right', fontsize='large')
    fig=plt.gcf()
    plt.show()
    fig.savefig("roc_plot_{}.png".format(filename)) if filename != None else None

    roc_auc_score(y_test, y_score)
    
    
###
def fi_plot(model, filename=None):
    fi=model.feature_importances_
    fi=pd.Series(model.feature_importances_, index=trs.columns)
    fig=fi.sort_values().iloc[-10:].plot('barh', title='Feature Importance').get_figure()#, figsize=(10,30))
    fig.savefig("fi_plot_{}.png".format(filename), bbox_inches='tight') if filename != None else None
    plt.show()
    
    
###
def cm_plot(y_test, y_pred, filename=None, legend_label = ['Neg', 'Pos']):
    cm=confusion_matrix(y_test, y_pred)

    # Normalize=True
    classes = legend_label

    print(cm, cm.sum(axis=1)[:, np.newaxis])

    cm_n=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    # Plot
    f, axs = plt.subplots(1,2, figsize=(10,8))
    axs=axs.flatten()
    plt.rcParams['axes.titlesize']='x-large'

    for img, ax in zip([cm_n, cm], axs):
        im=ax.imshow(img, interpolation='nearest', cmap=plt.cm.Blues)
        title="Confusion Matrix(Normalized)" if np.array_equal(img,cm_n) else "Confusion Matrix"

        ax.set(xticks=np.arange(img.shape[1]),
           yticks=np.arange(img.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
        
        ax.margins(y=0.4) # To fix truncated image, but how does this work?
        ax.set_xlabel('Predicted label', fontsize='x-large')
        ax.set_ylabel('True label', fontsize='x-large')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Add colorbar to axis.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, ax=ax, cax=cax)

        fmt = '.2f' if np.array_equal(img,cm_n) else 'd'
        thresh = img.max() / 2.

        for i, j in itertools.product(range(img.shape[0]), range(img.shape[1])):
            ax.text(j, i, format(img[i, j], fmt),
            horizontalalignment="center",
            color="white" if img[i, j] > thresh else "black", fontsize='x-large')

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    f.tight_layout()
    f.savefig("cm_plot_{}.png".format(filename)) if filename != None else None
    
    
### Plot learning curve for mri data
def plot_learning_curve(history, save_path, 
                        logs_to_plot=['loss', 'val_loss', 'Dec_GT_Output_dice_coefficient', 'val_Dec_GT_Output_dice_coefficient'],
                        initial_epoch=0):
    """
    === For SNUBH MRI Data ===
    plot learning curve and returns val_dice_max_epoch.
    
    
    Parameters
    ----------
    history : log.csv file path which was written by CsvLogger of Keras.
    """
    
    
    history = pd.read_csv(history)
    history['epoch'] = history['epoch'] + 1
    
    fig, axs = plt.subplots(1,2, figsize=(16,5))

    si=0
    ci=2
    key_3 = re.search(f"^([^\n_]+)", logs_to_plot[2]).group(1)
    
    for ax in axs:
    #     ax.set_xlim(1, len(history[logs_to_plot[0]]))
        for i in logs_to_plot[si:ci]:
            y_max=np.max(history[i])
            x_max=history['epoch'][np.argmax(history[i])]
            y_min=np.min(history[i])
            x_min=history['epoch'][np.argmin(history[i])]

            if re.search('val_loss', i) != None: print(f"val loss min value coordinate: ({x_min}, {y_min})")
            if re.search(f'val.+{key_3}', i) != None:
                print(f"val dice max value coordinate: ({x_max}, {y_max})")
                val_dice_max_epoch = x_max

            ax.plot(history['epoch'], history[i])
            ax.plot(x_max, y_max, 'co', label='_nolegend_') ; ax.plot(x_min, y_min, 'co', label='_nolegend_')
            ax.annotate(f'({x_max}, {y_max:.4f})', xy=(x_max, y_max), xytext=(x_max, y_max + 0.005))
            ax.annotate(f'({x_min}, {y_min:.4f})', xy=(x_min, y_min), xytext=(x_min, y_min - 0.005))

        ax.legend(logs_to_plot[si:ci], bbox_to_anchor=(1,0.5), loc='lower left')
        si += 2 ; ci +=2

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    
    return val_dice_max_epoch