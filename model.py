# Importing the libraries
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesRegressor


dataset = pd.read_csv('diabetes.csv', sep='\t')

X = dataset.iloc[:, :10]

y = dataset.iloc[:, -1]

regressor = ExtraTreesRegressor(n_estimators=100, random_state=0)

#Fitting model with trainig data
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[53, 1, 23.7, 92, 186, 109.2, 62, 3, 4.3041, 81]]))
