import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Color import Color
import sqlite3
import networkx as nx

# mmnist = input_data.read_data_sets("data", one_hot=True)
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

    return iris

def graph_clustering():
    df2 = df[df['Fraudulent_Claim'] == 'T']
    G = nx.from_pandas_edgelist(df2, "Name", "Service_Provider")
    pagerank_df = pagerank(G, df2)
    simrank_df = simrank(G, df2)
    hits_df = hits(G, df2)
    network_df = merge(df=df2, pagerank_df=pagerank_df, hits_df=hits_df)
    print("Network df: ")
    print(network_df)
    # triangle_count(G, df2)

def pagerank(G, df):
    pagerank_matrix = nx.pagerank(G,max_iter=200)
    arr = np.array(list(pagerank_matrix.values()))
    pagerank_matrix_df = pd.DataFrame(data=arr, columns=['Page_Rank'])
    # print("After page rank: ")
    # print(pagerank_matrix_df)
    return pagerank_matrix_df

def simrank(G, df):
    simrank_matrix = nx.simrank_similarity(G)
    arr = np.array(list(simrank_matrix.values()))
    simrank_matrix_df = pd.DataFrame(data=arr, columns=['Sim_Rank'])
    # print("After sim rank: ")
    # print(pagerank_matrix_df)
    return simrank_matrix_df

def hits(G, df):
    h, a = nx.hits(G)
    hubs = np.array(list(h.values()))
    authorities = np.array(list(a.values()))
    hits_arr = np.array(list(zip(hubs, authorities)))
    hits_df = pd.DataFrame(data=hits_arr, columns=['Hubs', 'Authorities'])
    # print(hits_df)
    return hits_df

def merge(df, pagerank_df, hits_df):
    merged_arr = np.array(list(zip(df['Name'], df['Service_Provider'], pagerank_df['Page_Rank'], hits_df['Hubs'], hits_df['Authorities'])))
    df = pd.DataFrame(data=merged_arr, columns=['Name', 'Service_Provider', 'Page_Rank', 'Hubs', 'Authorities'])
    return df

def triangle_count(G, df):
    trianglecount_matrix = nx.triangles(G)
    arr = np.array(list(trianglecount_matrix.values()))
    trianglecount_matrix_df = pd.DataFrame(data=arr, columns=['Triangle_Count'])
    print("After triangle count: ")
    print(trianglecount_matrix_df)
    print("Non zero: ")
    print(trianglecount_matrix_df[trianglecount_matrix_df['Triangle_Count'] != 0])

conn = sqlite3.connect('insurance.db')
print(Color.YELLOW + "Opening Database" + Color.END)
df = pd.read_sql_query("SELECT * FROM Claims",conn,  coerce_float=True, parse_dates=["Date_Of_Birth", "Policy_Start",
                                                 "Policy_End", "Date_Of_Loss", "Date_Of_Claim"])
graph_clustering()