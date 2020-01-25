# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from Color import Color
# import sqlite3
# import networkx as nx

# # mmnist = input_data.read_data_sets("data", one_hot=True)
# def load_data():
#     print(Color.YELLOW + "Opening Database" + Color.END)
#     conn = sqlite3.connect('insurance.db')
#     print(Color.YELLOW + Color.UNDERLINE + "Reading data ..." + Color.END)
#     df = pd.read_sql_query("SELECT * FROM Claims", conn, coerce_float=True,
#                            parse_dates=["Date_Of_Birth", "Policy_Start",
#                                         "Policy_End", "Date_Of_Loss", "Date_Of_Claim"])

#     iris = load_iris()



#     iris.feature_names = ['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount']
#     iris.target_names = ['Fraud','NOT Fraud']
#     tmp = df['Fraudulent_Claim'].astype(str).replace('*', 0)
#     tmp = tmp.astype(str).replace('', 1).astype(int)
#     iris.target = tmp.astype(int).values
#     # df = df[['Sum_Insured', 'Policies_Revenue', 'Broker_ID', 'Claim_Amount']]
#     iris.data = np.dstack( (df['Sum_Insured'].astype(np.float), df['Policies_Revenue'].astype(np.float),
#                             df['Broker_ID'].astype(np.int), df['Claim_Amount'].astype(np.float)) )[0]

#     return iris

# def graph_clustering():
#     edgelist = df[["callgno","calldno"]]
#     edgelist = edgelist.astype(str)
#     edgelist.to_csv("edgelist.csv",index=False)
#     G = nx.read_edgelist("edgelist.csv",delimiter=',')

# iris = load_data()