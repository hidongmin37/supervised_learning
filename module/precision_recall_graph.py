
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from sklearn.metrics import precision_recall_curve
import numpy as np

def precision_recall_curve_plot(y_test,pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도,재현율 nadrray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test,pred_proba_c1)
    
    # X축을 threshold값으로 Y축은 정밀도,재현율 값으로 각각 Plot 수행, 정밀도는 점선으로 표시
    
    plt.figure(figsize=(8,6))
    thresholds_boundary= thresholds.shape[0]
    plt.plot(thresholds,precisions[0:thresholds_boundary], linestyle='--', label='precision')
    plt.plot(thresholds,recalls[0:thresholds_boundary],label='recall')
    
    # threshold 값 X축의 Scale을 0.1단위로 변경
    
    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    
    # X축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
    
    