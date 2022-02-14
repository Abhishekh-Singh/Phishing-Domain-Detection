# from flask import Flask, request, jsonify, render_template
import pickle
# import pandas as pd
import numpy as np
import pandas as pd
from URLFeatureExtraction import featureExtraction

loaded_model = pickle.load(open("XGBoostClassifier.pkl", "rb"))

x = featureExtraction("http://www.example.com/index.html")


#converting the list to dataframe
feature_names = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']

data_frame = pd.DataFrame(data=[x],columns=feature_names) 
                    


# data1 = [np.array(data)]
prediction = loaded_model.predict(data_frame)
print(prediction)



## APP MAKING -----------

# app = Flask(__name__)
# model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [x for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     if prediction == 0:
#         output = "an Active"
#     elif prediction ==1:
#         output = "a Churn" 


#     return render_template('index.html', prediction_text='Customer would most probably  {} customer.'.format(output))


# if __name__ == "__main__":
#     app.run(debug=True)