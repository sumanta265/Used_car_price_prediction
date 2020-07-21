from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
app = Flask(__name__)
#model_load=pickle.load(open('car_price1.pkl','rb'))




@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")
span_new = pd.read_csv('sumanta_data1.csv')
span_X = span_new[['name',	'company',	'year','kms_driven',	'fuel_type'	]]
#Dummy Categorical Variables
span_X = pd.concat([span_X,pd.get_dummies(span_X['name'],prefix="name")],axis=1)
span_X= pd.concat([span_X,pd.get_dummies(span_X['company'],prefix="company")],axis=1)
span_X = pd.concat([span_X,pd.get_dummies(span_X['fuel_type'],prefix="fuel_type")],axis=1)

features_final = span_X.drop(columns=['name','company','fuel_type'])
X_train, X_test, y_train, y_test = train_test_split(features_final, span_new['Price'], test_size=0.33, random_state=42)

#nstatiate and Fit Model
rand_est = RandomForestRegressor()
rand_est.fit(X_train,y_train)

def prep_features(result):
    vector = pd.np.zeros(505) # Number of features in my dataset, dependent variables

    name1 = 'name_'+ result['name'] #Make it return the column name e.g Make_BMW
    model1 = 'company_'+ result['company']
    fuel1= 'fuel_type_' + result['fuel_type']


    name_index = features_final.columns.get_loc(str(name1)) # Get the index of the column using df.columns.get_loc()
    model_index = features_final.columns.get_loc(str(model1))
    fuel_index = features_final.columns.get_loc(str(fuel1))


    vector[0] = int(result['year'] )# Input values into your np.zeros vector
    vector[1] = int(result['kms_driven'])

    vector[name_index]= 1 # this will put a 1 at the right position ( depending on what make is selected)
    vector[model_index]= 1
    vector[fuel_index]= 1
    return vector



#FUNCTION that converts user's input into a vector that can be used by the model (29x features)

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    print(request.__dict__)
    if request.method == "POST":
        # name of company

        print(request.form.__dict__)  # this form will take the inputs
        company = request.form['company']

        model = request.form['model']
        year = request.form['year']
        fuel = request.form['fuel']
        km = request.form['km']

        result = dict(name=model, company=company, year=year,kms_driven=km,fuel_type=fuel)

        print("hey")
        test = prep_features(result)
        print(test)
        prediction = rand_est.predict([test])
        output=round(prediction[0],2)
        return render_template('index.html', prediction_text="Your car price is Rs. {}".format(output))

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
