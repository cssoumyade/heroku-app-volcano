import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
from model import *
from predict import *


def main():

    st.set_page_config(layout="wide")

    st.title("Volcanic Eruption Prediction from Seismic Signals")
    image_url = "https://images.unsplash.com/photo-1519901416153-b3ea11f3fcc6?ixlib=rb-1.2.1&ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=1350&q=80"
    st.image(image_url, width=800)

    st.sidebar.header("About")

    with open('about.txt', 'r') as f:
        about_txt = f.read()

    st.sidebar.write(about_txt)

    st.text("")
    st.text("")
    st.text("")

    exp = st.beta_expander("Instructions : ", expanded=True)
    with exp:
        "1. Please upload a csv file in the file uploader."
        "2. File should contain the 10-D sensor data."
        "3. Visualize the time series by toggling appropriate options."

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.subheader("File Uploader :")
    csv = st.file_uploader("Upload a segment file containing the seismic data from 10 sensors", type=['csv'])



    if csv is not None:
        st.text("")
        st.text("")
        st.text("")
        st.subheader("First few entries of the file : ")
        csv_file = pd.read_csv(csv)
        st.write(csv_file.head())
        pred = predict_time_to_erupt(csv_file)

        st.text("")
        st.text("")


        col_1, col_2 = st.beta_columns(2)
        col_2.subheader('Visualize')
        vizualize = col_2.radio('', ['No', 'Yes'])

        col_1.subheader("Time to eruption")
        col_1.text("{} centi-seconds".format(str(pred)))
        hm = ms_hm(pred)
        col_1.text('Which is approximately {} hours {} minutes'.format(hm['hours'], hm['minutes']))

        if vizualize=='Yes':
            with st.spinner(text='Please wait while we plot'):
                csv_file.plot(subplots=True, layout=(5,2), figsize=(20,10), title="Sensor data for the given segment")
                st.text("")
                st.text("")
                st.pyplot(plt)

if __name__=="__main__":
    main()
