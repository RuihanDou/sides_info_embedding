import numpy as np
import matplotlib.pyplot as plt



def plot_auc_trend(roc_auc: list, pr_auc: list, insert_eval_epoch: int):
    evals = len(roc_auc)
    assert len(roc_auc) == len(pr_auc)
    x_axis = np.array(0, evals * insert_eval_epoch, insert_eval_epoch).tolist()

    plt.plot(x_axis, roc_auc, 'b*--', alpha=0.5, linewidth=1, label='roc')
    plt.plot(x_axis, pr_auc, 'rs--', alpha=0.5, linewidth=1, label='pr')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.show()
