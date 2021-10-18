
from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
    send_file
)
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from io import BytesIO
from matplotlib.backends.backend_agg import  FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot  as plt
import jinja2
import pandas as pd
import seaborn as sns
from sklearn import metrics



import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

plt.style.use('ggplot')



app = Flask(__name__)
app.secret_key = 'somesecretkeythatonlyishouldknow'

model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

my_loader = jinja2.ChoiceLoader([
    app.jinja_loader,
    jinja2.FileSystemLoader('/Users/gaura/something'),
])
app.jinja_loader=my_loader


dfs = pd.read_csv("car data.csv")
df = pd.read_csv("car data.csv")
dfs.drop([86,196],axis=0,inplace=True)
df.drop([86,196],axis=0,inplace=True)
dfs['Price_Diff']=dfs['Present_Price']-dfs['Selling_Price']
dfs['Year_old'] = 2020 - dfs['Year']
df['Price_Diff']=df['Present_Price']-df['Selling_Price']





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

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=66)
#model = RandomForestRegressor()



@app.route('/plot6')
def plot6():
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    prediction = pd.DataFrame(predicts)
    R_2 = r2_score(y_test, model.predict(X_test))
    mean_sqare_E = mean_squared_error(y_test, model.predict(X_test))
    mean_abso_E = mean_absolute_error(y_test, model.predict(X_test))
    cv_mean = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error').mean()

    # Appending results to Lists
    r2.append(r2_score(y_test, model.predict(X_test)))
    cv_score.append(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean())
    mse.append(mean_squared_error(y_test, predicts))
    mae.append(mean_absolute_error(y_test, predicts))

    plt.style.use('ggplot')
    test_index=y_test.reset_index()["Selling_Price"]
    fig1=plt.figure()
    ax=test_index.plot(label="originals",figsize=(16,8),linewidth=2,color="r",marker='o',legend=True,title="ORIGINALS VS PREDICTIONS")
    ax=prediction[0].plot(label = "predictions",figsize=(16,8),linewidth=2,color="b",marker='*',legend=True)
    #fig1.legend(loc='upper right')
    #fig1.title("ORIGINALS VS PREDICTIONS")
    #fig1.xlabel("index")
    #fig1.ylabel("values")


    canvas=FigureCanvas(fig1)
    #ax.xlabel("index")
    #ax.ylabel("values")
    img1=BytesIO()
    fig1.savefig(img1)
    #fig1.close()
    img1.seek(0)
    #plt.show()
    return send_file(img1,mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/plot1')
def plot1():

    f, axes = plt.subplots(2, 2, figsize=(19, 8))
    sns.countplot(x='Transmission', data=dfs, ax=axes[1, 0])
    sns.countplot(x='Fuel_Type', data=dfs, ax=axes[1, 1])
    sns.countplot(x='Owner', data=dfs, ax=axes[0, 1])
    sns.countplot(x='Seller_Type', data=dfs, ax=axes[0, 0])
    canvas=FigureCanvas(f)
    img=BytesIO()
    f.savefig(img)
    img.seek(0)

    return send_file(img,mimetype='image/png')


@app.route('/plot2')
def plot2():
    f, axes = plt.subplots(2, 2, figsize=(19, 8))
    sns.barplot(x='Transmission', y='Price_Diff', data=dfs, ax=axes[1, 0])
    sns.barplot(x='Fuel_Type', y='Price_Diff', data=dfs, ax=axes[1, 1])
    sns.barplot(x='Owner', y='Price_Diff', data=dfs, ax=axes[0, 1])
    sns.barplot(x='Seller_Type', y='Price_Diff', data=dfs, ax=axes[0, 0])
    canvas=FigureCanvas(f)
    img=BytesIO()
    f.savefig(img)
    img.seek(0)

    return send_file(img,mimetype='image/png')


@app.route('/plot3')
def plot3():

    fig =plt.figure()
    sns.barplot(x='Year_old', y='Selling_Price', data=dfs)

    canvas=FigureCanvas(fig)
    img=BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img,mimetype='image/png')

@app.route('/plot4')
def plot4():
    data = pd.read_csv("car data.csv")
    corrmat = data.corr()
    top_corr_features = corrmat.index
    fig44=plt.figure(figsize=(10, 10))
    # plot heat map
    g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    canvas=FigureCanvas(fig44)
    img44=BytesIO()
    fig44.savefig(img44)
    img44.seek(0)

    return send_file(img44,mimetype='image/png')

@app.route('/plot5')
def plot5():

    fig5=plt.figure()
    plt.scatter(x='Price_Diff', y='Kms_Driven', data=dfs)
    canvas=FigureCanvas(fig5)
    img5=BytesIO()
    fig5.savefig(img5)
    img5.seek(0)

    return send_file(img5,mimetype='image/png')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        Year=2020-Year
        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0
        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0
        prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        output=round(prediction[0],2)


        if output<0:
            return render_template('prediction.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('prediction.html',prediction_text="You Can Sell The Car at {} Lakhs".format(output))
    else:
        return render_template('prediction.html')

if __name__=="__main__":
    app.run(debug=True)

