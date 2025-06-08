import mysql.connector 
import streamlit as st

#connection

conn=mysql.connector.connect(
    host='localhost',
    port='3306',
    user='root',
    passwd='',
    db='mydatabase'
    
    )
connect=conn.cursor()

def fetch_data():
    connect.execute('select * from Data order by id asc')
    data=connect.fetchall()
    return data 
