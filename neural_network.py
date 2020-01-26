#### Importing Libraries ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import keras
import sqlite3
from Color import Color
import sklearn
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import xgboost
import shap
import lime
import lime.lime_tabular
from lime import submodular_pick
import warnings
import lightgbm as lgb
import pydotplus
from datetime import date, datetime
import time
from IPython.core.display import display, HTML
import warnings

np.random.seed(2)

def load_data():
    print(Color.YELLOW + "Opening Database" + Color.END)
    conn = sqlite3.connect('insurance.db')
    print(Color.YELLOW + Color.UNDERLINE + "Reading data ..." + Color.END)
    df = pd.read_sql_query("SELECT * FROM Claims", conn, coerce_float=True,
                           parse_dates=["Date_Of_Birth", "Policy_Start",
                                        "Policy_End", "Date_Of_Loss", "Date_Of_Claim"])

    iris = load_iris()



    iris.feature_names = ['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount']
    iris.target_names = ['Fraud','NOT Fraud']
    tmp = df['Fraudulent_Claim'].astype(str).replace('*', 0)
    tmp = tmp.astype(str).replace('', 1).astype(int)
    iris.target = tmp.astype(int).values
    # df = df[['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount']]
    iris.data = np.dstack( (df['Sum_Insured'].astype(np.float), df['Policies_Revenue'].astype(np.float),
                            df['Broker_ID'].astype(np.int), df['Claim_Amount'].astype(np.float)) )[0]

    return iris, df

def decision_tree(train_data, train_target):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    print("Training set: " + str(len(train_target)))
    print("Testing set: " + str(len(test_target)))
    # print(clf.predict(test_data))
    pred = clf.predict(test_data)

    score = metrics.accuracy_score(test_target, pred)
    print("Accuracy: " + str(score*100.0))


    #viz code
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                        filled=True, rounded=True, impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("insurance.pdf")

## Randomforest
def random_forest(train_data, train_target):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                        criterion = 'entropy')
    classifier.fit(train_data, train_target)

    # Predicting Test Set
    y_pred = classifier.predict(test_data)
    acc = accuracy_score(test_target, y_pred)
    prec = precision_score(test_target, y_pred)
    rec = recall_score(test_target, y_pred)
    f1 = f1_score(test_target, y_pred)



    results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # results = results.append(model_results, ignore_index = True)

    print(results)

'''
Artifial Neural Network Model
'''
def ann(train_data, train_target):
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    columns = train_data.shape[1]

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units =30 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))

    # Adding the second hidden layer
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the third hidden layer
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(train_data,train_target, batch_size=32, epochs=10)

    # Predicting the Test set results
    y_pred = classifier.predict(test_data)
    y_pred = (y_pred > 0.5)

    score = classifier.evaluate(test_data, test_target)
    print(score)

def prob(data):
    
    # return np.array(list(zip(1-model.predict(data),model.predict(data))))
    return np.array(list(zip(1-model.predict(data),model.predict(data))))

def sp_lime(train_data, test_data, df):
    sp_obj = submodular_pick.SubmodularPick(explainer, train_data.values, \
    prob, num_features=5,num_exps_desired=10)
    [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]

def lime_pred(train_data, test_data, df):
    data = np.dstack( (df['Sum_Insured'].astype(np.float), df['Policies_Revenue'].astype(np.float),
                            df['Broker_ID'].astype(np.int), df['Claim_Amount'].astype(np.float)) )[0]
    explainer = lime.lime_tabular.LimeTabularExplainer(data,\
        feature_names=['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount'],\
        verbose=True, \
        mode='classification')

    lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':12,
    'min_data_in_leaf':100,
    'alpha':0.5}
    lgb_train = lgb.Dataset(train_data, train_target)
    lgb_eval = lgb.Dataset(train_data, train_target)

    model = lgb.train(lgb_params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)
    exp = explainer.explain_instance(data[0], lambda data: np.array(list(zip(1-model.predict(data),model.predict(data)))), num_features = 4)
    
    # display(HTML(exp).data)
    fig = exp.as_pyplot_figure()
    # fig.plot(range(0,100))
    plt.show()
    sp_obj = submodular_pick.SubmodularPick(explainer, data, \
    lambda data: np.array(list(zip(1-model.predict(data),model.predict(data)))),\
         num_features=4,num_exps_desired=10)
    explanations = [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]
    plt.show()

def shap_pred():
    model = xgboost.XGBClassifier().fit(train_data, train_target)

    # compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)
    columns = ['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount']
    # for i in range(4):
    #     for j in range(4):
    #         shap.dependence_plot(i, shap_values, train_data, feature_names=columns, interaction_index=columns[j])
    
    shap.summary_plot(shap_values, train_data,  feature_names = columns)

    shap.summary_plot(shap_values, train_data, feature_names = columns, plot_type='bar')

    # knn = sklearn.neighbors.KNeighborsRegressor()
    # knn.fit(train_data, train_target)

    # X_train_summary = shap.kmeans(train_data, 10)

    # t0 = time.time()
    # explainerKNN = shap.KernelExplainer(knn.predict,X_train_summary)
    # shap_values_KNN_test = explainerKNN.shap_values(test_data)
    # t1 = time.time()
    # timeit=t1-t0
    # timeit

    # for j in range(len(shap_values_KNN_test)):
    #     display(shap.force_plot(explainerKNN.expected_value, shap_values_KNN_test[j], test_data[[j]]))
    #     plt.show()

iris, df = load_data()

test_idx = [0, 1198]

train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

warnings.filterwarnings("ignore")

# decision_tree(train_data=train_data, train_target=train_target) 
# random_forest(train_data=train_data, train_target=train_target)
# ann(train_data=train_data, train_target=train_target)
# lime_pred(train_data=train_data, test_data=test_data,df=df[:1000])
shap_pred()

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# print(cm)

# #Let's see how our model performed
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

# ## EXTRA: Confusion Matrix
# cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
# df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
# plt.figure(figsize = (10,7))
# sn.set(font_scale=1.4)
# sn.heatmap(df_cm, annot=True, fmt='g')
# plt.savefig("ConfusionMatrix.png")
# print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

