🌾 Satellite-Based Crop Yield Prediction using Sentinel-2 & Deep Learning

📌 Overview

This project implements an end-to-end AI-driven pipeline to predict agricultural crop yield using Sentinel-2 satellite imagery and deep learning. The system extracts NDVI-based vegetation features from multispectral satellite data and trains a Convolutional Neural Network (CNN) to predict farm-level crop yield.

The solution combines remote sensing, computer vision, and machine learning to support precision agriculture and data-driven agritech decision-making.

🎯 Problem Statement

Traditional crop yield estimation relies on manual surveys and historical averages, which are time-consuming and often inaccurate. This project aims to automate crop yield prediction using satellite imagery and deep learning to provide scalable, accurate, and early forecasts.

🛰️ Data Source

Satellite: Sentinel-2 (European Space Agency)

Spatial Resolution: 10 meters

Revisit Frequency: ~5 days

Spectral Bands Used:

Band 4 – RED (Visible spectrum)

Band 8 – NIR (Near Infrared)

🌱 NDVI Feature Extraction

Vegetation health is quantified using the Normalized Difference Vegetation Index (NDVI).

NDVI Formula (Plain Text):

NDVI = (NIR − RED) / (NIR + RED)

Where:

NIR = Near Infrared band (Band 8)

RED = Red band (Band 4)

NDVI values range from -1 to +1:

Higher values indicate healthy vegetation

Lower values indicate stressed or sparse vegetation

NDVI values are calculated for each pixel, masked using farm boundaries, and time-stamped for temporal modeling.

🔍 Data Acquisition

Farm boundaries are defined using GeoJSON files

Sentinel-2 imagery is fetched using the Sentinelsat (Copernicus) API

Relevant spectral bands are downloaded automatically

Data is stored locally for preprocessing

🧪 Preprocessing Pipeline

Crop satellite images to farm boundaries

Extract RED and NIR spectral bands

Compute NDVI values

Normalize NDVI values for model input

Stack multi-temporal NDVI data into structured tensors

🤖 Model Architecture

A custom deep learning regression model is built using TensorFlow.

Model Details:

Input: NDVI-based time-series tensors

Three convolutional blocks:

Convolution layer with ReLU activation

Batch normalization

Dropout for regularization

Fully connected (dense) layers

Output: Continuous crop yield value

Training Configuration:

Loss Function: Mean Squared Error (MSE)

MSE Formula (Plain Text):

MSE = (1 / n) × Σ (actual_yield − predicted_yield)²

Optimizer: Adam

Regularization: Dropout and L2 weight decay

📈 Results & Performance

Processed 50+ NDVI tensors from multi-temporal Sentinel-2 imagery

Achieved high correlation (~0.82) between predicted and actual crop yields

Validation loss showed smooth convergence during training

NDVI trends aligned well with known seasonal crop growth cycles

🛠️ Technologies Used
Programming Language

Python

Libraries & Frameworks

TensorFlow / Keras

NumPy

OpenCV

GDAL

Rasterio

Matplotlib

Tools & APIs

Sentinelsat (Copernicus API)

GeoJSON

Python standard libraries

🚀 How to Run

Download Sentinel-2 satellite data:

python Sentinel_data_scraper.py


Preprocess images and compute NDVI:

python Cropping_preprocessing.py


Train the deep learning model:

python Model_TF.py


Evaluate predictions and visualize results

🌍 Real-World Applications

Precision agriculture and smart farming

Seasonal crop yield forecasting

Government agricultural planning

Crop insurance risk assessment

Food supply chain optimization

🔮 Future Improvements

Integrate weather and soil moisture data

Improve temporal modeling using advanced architectures

Deploy as a web-based decision support system

Extend the model to multiple crops and regions

📄 Summary

This project presents a complete AI-driven pipeline that converts raw Sentinel-2 satellite imagery into accurate, farm-level crop yield predictions. By automating data acquisition, NDVI feature extraction, and deep learning-based modeling, it supports smarter agricultural planning and scalable agritech solutions.
