from flask import Flask ,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model//Modelscaler.pkl", 'rb'))
model = pickle.load(open("Model//log_reg", 'rb'))

##Route the homepage
@app.route('/')
def index():
    return "<h1>Welcome to home page</h1>"

##Route for single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method =='POST':

        Pregnancies =int(request.form.get("Pregnancies"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure =float(request.form.get("BloodPressure"))
        SkinThinkness =float(request.form.get("SkinThinkness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))


        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,
                                      SkinThinkness,Insulin,BMI,DiabetesPedigreeFunction,
                                      Age]])
        predict = model.predict(new_data)

        if predict[0]==1:
            result = "Diabetic"
        else:
            result = "Normal,No Diabetic"

        return render_template("single_prediction.html",result=result)

    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)
