
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import numpy as np

def roc_curve_plot(y_test, pred_proba_c1):

    # 임곗값에 따른 FPR, TPR 값을 반환 받음
    fprs, tprs, thresholds = roc_curve(y_test,pred_proba_c1)
    plt.plot(fprs,tprs,label='Random')
    # ROC Curve를 plot 곡선으로 그림
    
    plt.plot([0,1],[0,1], 'k--', label = 'Random')
    start, end =plt.xlim()
    # 가운데 대각선 직선을 그림
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR(1-Sensitivity)'); plt.ylabel('TPR(Recall)')
    plt.legend()
    plt.show()
    
    
    