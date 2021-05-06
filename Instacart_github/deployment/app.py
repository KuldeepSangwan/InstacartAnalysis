from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from tqdm import tqdm
import sklearn.metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.externals import joblib
from F1Optimizer import F1Optimizer
import random
import time
from prediction import Prediction



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

@app.route('/')
def index():
    
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form['user_id']
    print(user_id)
    pred=[]
    frst1 = time.time()
    dow = datetime.now().month
    hour = datetime.now().hour
    user_id = int(user_id)
    her = here1[here1['user_id']==user_id]
    if her.shape[0]!=0:
        
        def get_order_dow(x):
            if any(x['order_dow']==dow):
                return x[x['order_dow']==dow]
            else:
                a = x[x['user_id']==user_id].head(1)
                a['order_dow']=dow
                a['week_product_reordered']=0
                a['week_product_ordered']=0
                a['week_product_reordered_ration']=0
                return a

        am = her.groupby('product_id').apply(get_order_dow)
        convert_dict = {'user_id': int,
                        'product_id': int,
                        'order_dow': int,
                        'week_product_reordered': int,
                        'week_product_ordered': int,
                        'week_product_reordered_ration': float,
                       }
        data = am.astype(convert_dict)
        data.reset_index(drop=True, inplace=True) 
        data = data.merge(list_days,on=["user_id", 'product_id'])
        # del list_days
        data = data.merge(products,on="product_id")
        data = data.merge(orders_prior_department_reordered,on=['user_id', 'department_id'])
        data = data.merge(orders_prior_aisle_reordered,on=['user_id', 'aisle_id'])

        data = data.merge(productFeat,on='product_id')
        data = data.merge(user_feat1,on='user_id')
        data = data.merge(product_features1,on='product_id')
        data = data.merge(user_product_order,on=['user_id','product_id'])
        data = data.merge(temp_data,on=['user_id','product_id'])
        data = data.merge(product_embeddings,on='product_id')
        print("4")

        # gave random value for some of the parameters
        data['days_since_prior_order']=float(random.randint(1, 30))
        data['order_number']=random.randint(10, 20)
        data['order_hour_of_day']=hour

        test = data[['product_id', 'user_id', 'order_number', 'order_dow',
           'order_hour_of_day', 'days_since_prior_order', 'aisle_id',
           'department_id', 'unique_prod_in_aisle', 'aisle_product_reordered',
           'unique_prod_in_department', 'department_product_reordered',
           'number_of_unique_users_for_product',
           'number_of_unique_users_for_product_reordered', '0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
           '30', '31', 'days_before_product_ordered-1',
           'days_before_product_ordered-4', 'days_before_product_ordered-2',
           'days_before_product_ordered_mean', 'days_before_product_ordered-3',
           'days_before_product_ordered_median', 'prod_orders',
           'prod_ordersBYtotal_orders', 'product_ordered_last',
           'product_ordered_first', 'prod_min_orderNumber', 'prod_max_orderNumber',
           'prod_cart_mean', 'days_since_prior_order_mean', 'order_dow_mean',
           'order_hour_of_day_mean', 'add_to_cart_order_inverse_mean',
           'add_to_cart_order_relative_mean', 'reordered_sum',
           'week_product_reordered', 'week_product_ordered',
           'week_product_reordered_ration', 'user_orders_count', 'user_since',
           'user_mean_days_since_prior_order',
           'user_median_days_since_prior_order', 'unique_prod',
           'NumberTimeProductsOrdered', 'user_reorder_ratio',
           'Avergae_Basket_size']]

        bst = joblib.load('final_sub/model.pkl') 
        pred_Xcv = bst.predict(xgb.DMatrix(test))

        data['pred'] = pred_Xcv
        test_pred = data[['user_id','product_name','product_id','pred']].groupby('user_id').agg({'product_id': list,'product_name': list, 'pred': list})

        def sort_values(row):  
            ok = np.argsort(np.array(row['pred']))
            test_pred.at[row.name,'product_id']=[row['product_id'][i] for i in ok][::-1]
            test_pred.at[row.name,'product_name']=[row['product_name'][i] for i in ok][::-1]
            test_pred.at[row.name,'pred'] = [row['pred'][i] for i in ok][::-1]

        test_pred.apply(sort_values,axis=1)

        for index, row in test_pred.iterrows():
            opt=F1Optimizer.maximize_expectation(row.pred)
            best_k=opt[0]
            pred=test_pred[test_pred.index==index]['product_name'].values[0][:best_k]
    else:
        Top_prod_DOW_Hour = joblib.load('final_sub/Top_prod_DOW_Hour.pkl')
        pred = Top_prod_DOW_Hour[(Top_prod_DOW_Hour['order_dow']==dow) & (Top_prod_DOW_Hour['order_hour_of_day']==hour)]['top_prod'].values[0]
        if not pred:
            Top_prod_Hour = joblib.load('final_sub/Top_prod_Hour.pkl')
            pred = Top_prod_Hour[Top_prod_DOW_Hour['order_hour_of_day']==hour]['top_prod'].values[0]
    tm = time.time() - frst1
    print('Time it took to Predct(sec): ', tm)
    print(pred,tm)
    if len(pred)>0: 
        return flask.render_template("product_list.html",predictions = pred)
    return jsonify({'Prediction': pred})


if __name__ == '__main__':
    here1 = joblib.load('final_sub/user_prod_orderdow.pkl')
    list_days = joblib.load('final_sub/list_days.pkl')
    products = joblib.load('final_sub/products.pkl')
    orders_prior_department_reordered = joblib.load('final_sub/orders_prior_department_reordered.pkl')
    orders_prior_aisle_reordered = joblib.load('final_sub/orders_prior_aisle_reordered.pkl')
    productFeat = joblib.load('final_sub/productFeat.pkl')
    user_feat1 = joblib.load('final_sub/user_feat1.pkl')
    product_features1 = joblib.load('final_sub/product_features1.pkl')
    user_product_order = joblib.load('final_sub/user_product_order.pkl')
    temp_data = joblib.load('final_sub/order_prod_addToCart.pkl')
    product_embeddings = joblib.load('final_sub/product_embeddings.pkl')
    app.run(host='127.0.0.1', port=8080)
