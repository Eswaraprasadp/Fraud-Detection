from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap
from flask_moment import Moment
import jinja2
import jwt
import json
import traceback
import pandas as pd
import sqlite3
from Color import Color

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)
moment = Moment(app)
# jwt = JWTManager(app)

print(Color.YELLOW + "Opening Database"+ Color.END)
conn = sqlite3.connect('insurance.db')
print(Color.YELLOW + Color.UNDERLINE+ "Reading data ..."+ Color.END)
df = pd.read_sql_query("SELECT * FROM Claims",conn,  coerce_float=True, parse_dates=["Date_Of_Birth", "Policy_Start",
                                                 "Policy_End", "Date_Of_Loss", "Date_Of_Claim"])

from application import routes