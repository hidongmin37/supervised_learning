from sklearn.preprocessing import Binarizer


from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score,confusion_matrix,roc_auc_score


def get_clf_eval(y_test, pred,pred_proba):
    confusion = confusion_matrix(y_test,pred)
    accuracy =accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    roc_score=roc_auc_score(y_test,pred_proba)
    
    # pred=lr_clf.predict(X_test)
    # roc_score = roc_auc_score(y_test,pred)
    
    
    
    # F1 스코어 추가
    f1 = f1_score(y_test,pred)
    print('오차 행렬')
    print(confusion)
    
    # f1 score print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율 :{2:.4f} ,F1 : {3:.4f} , ROC AUC 값 : {4:.4f}'.format(accuracy,precision,recall,f1,roc_score))
    
def get_eval_by_threshold(y_test,pred_proba_c1,threshold):
    
    for custom_threshold in threshold:
        # thresholds 리스트 객체내의 값을 차례로 iteration하면서 Evaluation 수행
        binarizer= Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:' ,custom_threshold)
        get_clf_eval(y_test,custom_predict,pred_proba_c1)
        