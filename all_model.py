
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

#scoring and tuning 
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import r2_score


df = pd.read_csv("D:\\TECMPN\\MiniProject\\Car data.csv")




df['Price_Diff']=df['Present_Price']-df['Selling_Price']







df['Year_old']=2019-df['Year']






df.drop([86,196],axis=0,inplace=True)



fuel_dummies =pd.get_dummies(df[['Fuel_Type','Transmission','Seller_Type']], drop_first=True)

df  = pd.concat([df,fuel_dummies],axis = 1)
df.drop(['Car_Name','Fuel_Type','Transmission','Seller_Type','Price_Diff'],axis=1,inplace=True)


cv=5
r2=[]
cv_score=[]
mae=[]
mse=[]

X=df.drop('Selling_Price',axis=1)

y=df['Selling_Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=66)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def results(model,X_train,X_test,y_train,y_test): 
    model.fit(X_train,y_train)
    predicts=model.predict(X_test)
    prediction=pd.DataFrame(predicts)
    R_2=r2_score(y_test,model.predict(X_test))
    mean_sqare_E =mean_squared_error(y_test,model.predict(X_test))
    mean_abso_E =mean_absolute_error(y_test,model.predict(X_test))
    cv_mean = -cross_val_score(model,X_train,y_train,cv=cv, scoring='neg_mean_squared_error').mean()
    
    # Appending results to Lists 
    r2.append(r2_score(y_test,model.predict(X_test)))
    cv_score.append(-cross_val_score(model,X,y,cv=cv, scoring='neg_mean_squared_error').mean())
    mse.append(mean_squared_error(y_test,predicts))
    mae.append(mean_absolute_error(y_test,predicts))
    
    # Printing results  
    print(model,"\n") 
    print("r^2 value :",R_2,"\n")
    print('mean square error',mean_sqare_E,"\n")
    print('mean absolute error',mean_abso_E,"\n")
    print("CV score:",cv_mean,"\n")
    print('#'*40)
    # Plot for prediction vs originals
    plt.style.use('ggplot')
    test_index=y_test.reset_index()["Selling_Price"]
    ax=test_index.plot(label="originals",figsize=(16,8),linewidth=2,color="r",marker='o')
    ax=prediction[0].plot(label = "predictions",figsize=(16,8),linewidth=2,color="b",marker='*')
    plt.legend(loc='upper right')
    plt.title("ORIGINALS VS PREDICTIONS")
    plt.xlabel("index")
    plt.ylabel("values")
    plt.show()





"""
model=ExtraTreesRegressor()
model.fit(X,y)
#print(model.fit(x,y))
#print(model.feature_importances_)

feat_imp=pd.Series(model.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='barh')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
#print(X_train)
#print(X_train.shape)

regressor=RandomForestRegressor()

n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#print(n_estimators)

max_features=['auto','sqrt']

max_depth=[ int(x) for x in np.linspace(5,30,num=6)]

min_samples_split= [2,5,10,15,100]

min_samples_leaf= [1,2,5,10]
random_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}



n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#print(n_estimators)

max_features=['auto','sqrt']

max_depth=[ int(x) for x in np.linspace(5,30,num=6)]

min_samples_split= [2,5,10,15,100]

min_samples_leaf= [1,2,5,10]
random_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

#print(random_grid)

rf=RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

"""

rfr = RandomForestRegressor()
results(rfr,X_train,X_test,y_train,y_test)


lg = LinearRegression()
results(lg,X_train,X_test,y_train,y_test)




dtr =DecisionTreeRegressor()
results(dtr,X_train,X_test,y_train,y_test)



params = {"alpha": [.01, .1, .5, .7, 1, 1.5, 2, 2.5, 3, 5, 8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,30]}
ridreg = Ridge()
clf = GridSearchCV(estimator=ridreg, param_grid=params, cv=5, return_train_score=True)
clf.fit(X_train, y_train)

#print(clf.best_params_)

results(clf,X_train,X_test,y_train,y_test)


params = {"alpha": [.00001, .0001, .001, .005, .01, .1, 1, 5]}
lasreg = Lasso()
clf = GridSearchCV(estimator=lasreg, param_grid=params, cv=5, return_train_score=True)
clf.fit(X_train, y_train)

#print(clf.best_params_)

results(clf,X_train,X_test,y_train,y_test)

Results = pd.DataFrame({
    'model':['random Forest','linear','Dicision Tree','Ridge','Lasso'],
    'r^2':r2,
    'cv_score':cv_score,
    'mae':mae,
    'mse':mse
})

print(Results)