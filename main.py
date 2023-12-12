from streamlit_option_menu import option_menu
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


st.header("Klasifikasi Artikel Berita Dengan Spark", divider='rainbow')
new_text = [st.text_area("Masukkan Artikel Berita")]

button = st.button("Submit")

if button:
    label = {0.0 : "Edu", 1.0 : "Sport", 2.0 : "Food"}
    # Inisialisasi sesi Spark
    spark = SparkSession.builder.appName("KlasifikasiBerita").getOrCreate()

    # Load the saved model
    model = PipelineModel.load("model")
    
    # membuat dataframe spark
    df = spark.createDataFrame([(text,) for text in new_text], ["stopwords"])
    
    # Make predictions on the new data
    predictions = model.transform(df).select("prediction").collect()[0].prediction
    predict_result = label[predictions]
    st.write(f"Hasil Prediksi : {predict_result}")