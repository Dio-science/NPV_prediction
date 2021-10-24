
from flask import Flask, render_template, request
import numpy as np
import pickle
from flask_jsonpify import jsonify
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)


model = pickle.load(open('ExtraNPVmodel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    period = float(int_features[2])
    Dis_rate = float(int_features[0])
    Undis_CF = float(int_features[1])
    Dis_rate = Dis_rate/100
    Dis_factor = 1/(1 + Dis_rate)**period
    Pre_Val= Undis_CF*Dis_factor
    Dis_val = Undis_CF - Pre_Val
    Dis_rate = Dis_rate*100
    values = [period, Dis_rate, Dis_factor, Undis_CF, Pre_Val, Dis_val]
    final_features = [np.array(values)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    period, Dis_rate, Dis_factor, Undis_CF, Pre_Val, Dis_val = round(period), round(Dis_rate, 2), round(Dis_factor, 2), round(Undis_CF, 2), round(Pre_Val, 2), round(Dis_val, 2)

    return render_template('index.html', prediction_text='With a cash flow of ${} through a period of {} years, and a discount rate of {}%. The estimated discount factor is {}, discounted value of ${}, and a present value of ${}. The Predicted NPV over a period of {} years is ${}'.format(Undis_CF,period,Dis_rate,Dis_factor,Dis_val,Pre_Val,period,output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
