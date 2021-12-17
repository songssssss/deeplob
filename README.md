# deeplob


print(confusion_matrix(ConY, ConX))
'''
[[27684  6886  3837]
 [ 5214 56485  4297]
 [ 4782  6429 23873]]
'''

print(classification_report(ConY, ConX))
'''
              precision    recall  f1-score   support

         0.0       0.73      0.72      0.73     38407
         1.0       0.81      0.86      0.83     65996
         2.0       0.75      0.68      0.71     35084

    accuracy                           0.77    139487
   macro avg       0.76      0.75      0.76    139487
weighted avg       0.77      0.77      0.77    139487

'''
