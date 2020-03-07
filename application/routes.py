from application import app, df
from flask import render_template, flash, redirect, url_for, request, abort
from datetime import datetime
import os
from neural_network import load_data, decision_tree, random_forest, ann, shap_pred, lime_pred

@app.route('/', methods=['GET', 'POST'])
def index():
    return "Hello world"

@app.route('/db', methods=['GET'])
def show_db():
    columns = list(df.columns)
    print("df columns", columns)

    values = [[str(v) for v in df[:10].values[i]] for i in range(10)]
    print("Values: ", values)
    return render_template('table.html', columns=columns, values=values)

@app.route('/neural', methods=['GET'])
def run_neural():
    return 

@app.route('/shap', methods =['GET'])
def shap_plots():
    files = []
    summary_images = []
    feature_names = ['Sum Insured', 'Policies Revenue', 'Broker ID', 'Claim Amount']
    summary_images_names = ['summary_plot.png', 'summary_plot_bar.png']

    file_names = ['1_2.png', '1_3.png', '1_4.png', 
    '2_1.png', '2_3.png', '2_4.png', 
    '3_1.png', '3_2.png', '3_4.png', 
    '4_1.png', '4_2.png', '4_3.png', ]
    for name in file_names:
        f = '/static/upload/' + name
        files.append(f)

    for name in summary_images_names:
        f = '/static/upload/' + name
        summary_images.append(f)

    return render_template('shap.html', files = files, feature_names = feature_names, summary_images=summary_images)

if __name__=='__main__':
    app.run(debug=True)