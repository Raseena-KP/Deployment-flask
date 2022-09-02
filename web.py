from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
        return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features).astype(int)
        output=prediction.item()    
        
        
        out_arr={'No risk of coronary heart disease ': 0, 'There is risk of coronary heart disease ': 1}
        output=list(out_arr.keys())[list(out_arr.values()).index(output)]
        
        return render_template("result.html", prediction_text=output)

if __name__=="__main__":
    app.run(port=5000)