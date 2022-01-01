# deeplob


print(confusion_matrix(ConY, ConX))
'''

    [27684  6886  3837]
    [ 5214 56485  4297]
    [ 4782  6429 23873]

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
    
    
    
https://blog.csdn.net/weixin_44863328/article/details/107303773

# interrupt training when the validation loss isn't decreasing anymore?

use an EarlyStopping callback:

```
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```
https://keras.io/getting_started/faq/#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore

# Accuracy Decreasing with higher epochs

https://stackoverflow.com/questions/53242875/accuracy-decreasing-with-higher-epochs


