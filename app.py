from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('model.pkl')
onehot = joblib.load('OneHotee.joblib')


@app.route('/')
@app.route('/home')
def main():
	return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	c = ["Fuel_Type","Seller_Type","Transmission","Present_Price","Kms_Driven","Owner","No_year"]
	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df.iloc[:,:3])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,3:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	print(int_features)

	return render_template("home.html",prediction_text="Car Price is : {}".format(result))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port =7000)