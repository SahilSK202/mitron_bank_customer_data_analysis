import numpy as np
import joblib

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

filename = 'STD_LR_model.joblib'
loaded_model = joblib.load(filename)

data = [69308,29267,13698,42.23, 46.80, 1,0,1,4,0]
data = np.array(data).reshape(1, -1)
pred = loaded_model.predict(data)
print(pred.item())
print(max(loaded_model.predict_proba(data)[0]))
print((loaded_model.predict_proba(data)[0]))