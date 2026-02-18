import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Telcom Data Dashboard",
    page_icon=":bar_chart:",
    layout='wide'
)
st.title('Telcom Dashboard')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

def load_file(file):
    if file.type == 'text/csv' or file.type == 'text/plain':
        return pd.read_csv(file, encoding="ISO-8859-1")
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return pd.read_excel(file)
    else:
        return None

f1 = st.file_uploader(":file_folder: Upload a file", type=["csv", "text", "xlsx"])

if f1 is not None:
    df = load_file(f1)
else:
    try:
        df = pd.read_csv('/path/to/cleaned_dataset_1.csv', encoding="ISO-8859-1")
    except FileNotFoundError:
        st.error("Default dataset not found. Please upload a file.")
        st.stop()

col1, col2 = st.columns((2))
df['StartDate'] = pd.to_datetime(df['StartDate'])
df['EndDate'] = pd.to_datetime(df['EndDate'])
Start_time = df['StartDate'].max()
End_time = df['EndDate'].max()
with col1:
    date1 = pd.to_datetime(st.date_input("Start date", Start_time))
with col2:
    date2 = pd.to_datetime(st.date_input("End date", End_time))
df_filtered = df[(df['StartDate'] >= date1) & (df['EndDate'] <= date2)].copy()

handset_manufacturer = st.sidebar.multiselect('Pick the Handset Manufacturer', df['Handset Manufacturer'].unique())
if not handset_manufacturer:
    df2 = df_filtered.copy()
else:
    df2 = df_filtered[df_filtered['Handset Manufacturer'].isin(handset_manufacturer)]

handset_type = st.sidebar.multiselect('Pick the Handset Type', df2['Handset Type'].unique())
if not handset_type:
    df3 = df2.copy()
else:
    df3 = df2[df2['Handset Type'].isin(handset_type)]

msisdn = st.sidebar.multiselect('Pick the MSISDN', df3['MSISDN/Number'].unique())
if msisdn:
    filtered_df = df3[df3['MSISDN/Number'].isin(msisdn)]
else:
    filtered_df = df3

category_df = filtered_df.groupby(by=['Netflix UL (Bytes)'], as_index=False)['Netflix DL (Bytes)'].count()
with col1:
    st.subheader('Netflix UL vs DL')
    fig = px.bar(category_df, x='Netflix UL (Bytes)', y='Netflix DL (Bytes)', template='seaborn')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Total UL and DL (Bytes customers)')
    fig = px.pie(filtered_df, values='Total UL (Bytes)', names='Handset Manufacturer', hole=0.5)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

cl1, cl2 = st.columns(2)
with cl1:
    with st.expander('Netflix Data View'):
        st.write(category_df.style.background_gradient(cmap='viridis'))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name='NetflixData.csv', mime='text/csv', help='Click here to download the data as csv file')

with cl2:
    with st.expander('Total UL Data by Handset Manufacturer'):
        handset_manufacturer_df = filtered_df.groupby(by='Handset Manufacturer', as_index=False)['Google UL (Bytes)'].sum()
        st.write(handset_manufacturer_df.style.background_gradient(cmap='viridis'))
        csv = handset_manufacturer_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name='HandsetManufacturer.csv', mime='text/csv', help='Click here to download the data as csv file')

# Ensure matplotlib is installed
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("matplotlib is required for background_gradient to work.")
