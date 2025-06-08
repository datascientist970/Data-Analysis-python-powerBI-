import streamlit as st 
import pandas as pd 
import plotly.express as px
from query import *
import time 

from streamlit_option_menu import option_menu

from numerize.numerize import numerize 

st.set_page_config(page_title="Dashboard",page_icon="📊",layout="wide")
st.subheader("Analytical Dashboard")
st.markdown("##")


result=fetch_data()

df=pd.DataFrame(result,columns=['id','Policy','Expiry','Location','State','Region','Investment','Construction','BusinessType','Earthquake','Flood','Rating'])
#st.dataframe(df)

st.sidebar.image('logo.png',caption='online analytics')


st.sidebar.header("Filter")

reg=st.sidebar.multiselect("Select Region", options=df['Region'].unique(),default=df["Region"].unique())

loc=st.sidebar.multiselect("Select Location", options=df['Location'].unique(),default=df["Location"].unique())

con=st.sidebar.multiselect("Select Construction", options=df['Construction'].unique(),default=df["Construction"].unique())

selection=df.query(
    "Region==@reg & Location==@loc & Construction==@con")

#st.dataframe(selection)

def Home():
    

    #with st.expander('Tabular'):
     #   show = st.multiselect("Filter:", options=selection.columns, default=[])
     #   st.write(selection[show])
   
    
    total_income = selection["Investment"].sum()  
    income_mode =selection["Investment"].mode()  
    #income_mode = income_mode_series.iloc[0] if not income_mode_series.empty else 0
    income_mean = selection["Investment"].mean()  
    income_median =selection["Investment"].median()  
    rating = selection['Rating'].sum()  

    
    total1, total2, total3, total4, total5 = st.columns(5, gap="large")
    
    with total1:
        st.info("Total Income:")
        st.metric(label="Sum USD", value=f"{total_income:}")
    
    with total2:
        st.info("Most Frequently:")
        st.metric(label="Mode USD", value=f"{income_mode:}")
    
    with total3:
        st.info("Average:")
        st.metric(label="Average USD", value=f"{income_mean:}")
    
    with total4:
        st.info("Central Earning:")
        st.metric(label="Median USD", value=f"{income_median:}")
    
    with total5:
        st.info("Rating:")
        st.metric(label="Rating", value=numerize(rating),help="Total Rating: {rating}")

    st.markdown("----")


Home()


#Graph

def Progressbar():
    st.markdown("""<style>.stProgress >div >div >div >div {background-image:linear-gradient(to right, #99ff99, #FFFF00)}</style>""",unsafe_allow_html=True)
    target=12000000000
    current=selection["Investment"].sum()
    percentage=round((current/target)*100)
    bar=st.progress(0)
    
    if percentage>100:
        st.subheader("Taget Done!")
    
    else:
        if percentage>20:
            html_content = f"""<p>Total of <span style="font-size:24px; color:yellow;">{percentage}</span> %</p>"""
   
            st.write(html_content, unsafe_allow_html=True)
            for i in range(percentage):
        
                time.sleep(0.1)
                bar.progress(i+1,text=" Target Percentage")
        elif percentage<10:
            html_content = f"""<p>Total of <span style="font-size:24px; color:red;">{percentage}</span> %</p>"""
   
            st.write(html_content, unsafe_allow_html=True)
            for i in range(percentage):
        
                time.sleep(0.1)
                bar.progress(i+1,text=" Target Percentage")
        elif percentage>50 or percentage<70:
            html_content = f"""<p>Total of <span style="font-size:24px; color:blue;">{percentage}</span> %</p>"""
   
            st.write(html_content, unsafe_allow_html=True)
            for i in range(percentage):
        
                time.sleep(0.1)
                bar.progress(i+1,text=" Target Percentage")
            
        elif percentage>70 or percentage<100:
            html_content = f"""<p>Total of <span style="font-size:24px; color:green;">{percentage}</span> %</p>"""
   
            st.write(html_content, unsafe_allow_html=True)
            for i in range(percentage):
        
                time.sleep(0.1)
                bar.progress(i+1,text=" Target Percentage")
            
       
Progressbar()


def Graph():
    total_investment=selection['Investment'].sum()
    rating=selection['Rating'].mean()
    business=(
        selection.groupby(by=['BusinessType']).count()[["Investment"]].sort_values(by="Investment")
        
        )
    fig=px.bar(
        business,x='Investment',y=business.index,
        orientation="h", 
        title="<b> Investment By Business Type</b>",
        color_discrete_sequence=["#0083b8"]*len(business),
        template="plotly_white"
        )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
        )
    
    state=selection.groupby(by=['State']).count()[["Investment"]]
        
        
    
    fig2=px.line(
        state,x=state.index,y='Investment',
        orientation="v", 
        title="<b> Investment By State</b>",
        color_discrete_sequence=["#0083b8"]*len(state),
        template="plotly_white"
        )
    fig2.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False))
        )
    left,right=st.columns(2)
    left.plotly_chart(fig2,use_container_width=True)
    right.plotly_chart(fig,use_container_width=True)
    
Graph()

