import xgboost as xgb
import pandas as pd
import numpy as np
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('cust_conv_prediction.json')

input_col = ['Avg Weekly clicks','ENT','CB_Y']
input_dict = dict()
data_clicks = input('Enter the number of clicks (comma seperated): ').split(',')
#print(data_clicks)
clicks = [int(x) for x in data_clicks]
input_dict['Avg Weekly clicks'] = [sum(clicks)/len(clicks)]
input_dict['ENT'] = [int(input('Enter 1 for enterprise account, 0 for small business: '))]
input_dict['CB_Y'] = [int(input('Enter 1 if chatbot used, 0 for chatbot not used: '))]

X_test = pd.DataFrame(input_dict)
prediction = loaded_model.predict(X_test)
if(prediction):
    print("The customer would convert")
else:
    print("The customer would not  convert")