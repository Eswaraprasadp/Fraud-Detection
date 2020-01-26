from application import app, df
from flask import render_template, flash, redirect, url_for, request, abort
from datetime import datetime
import os

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

@app.route('/shap', methods =['GET'])
def shap_plots():
    files = ['1_2.png', '1_3.png', '1_4.png', '2_1.png', '2_3.png', '2_4.png' ]
    file_names = ['1_2.png', '2_3.png']
    for name in file_names:
        f = 'http:127.0.0.1:5000/static/upload/' + name
        file_names.append(f)

    return render_template('shap.html', file_names = file_names)

if __name__=='__main__':
    app.run(debug=True)