🌾 Satellite-Based Crop Yield Prediction using Sentinel-2 & Deep Learning
📌 Overview

This project implements an end-to-end AI-driven pipeline to predict agricultural crop yield using Sentinel-2 satellite imagery and deep learning. The system extracts NDVI-based vegetation features from multispectral satellite data and trains a Convolutional Neural Network (CNN) to accurately predict farm-level crop yield.

The solution combines remote sensing, computer vision, and machine learning to support precision agriculture, yield forecasting, and agritech decision-making.

🎯 Problem Statement

Traditional crop yield estimation relies on manual surveys and historical averages, which are time-consuming, expensive, and often inaccurate. This project aims to:

Automate crop monitoring using satellite imagery

Quantify vegetation health using NDVI

Predict crop yield ahead of harvest using deep learning

Build a scalable and location-independent solution

🛰️ Data Source

Satellite: Sentinel-2 (European Space Agency)

Spatial Resolution: 10 meters

Revisit Frequency: ~5 days

Spectral Bands Used:

Band 4 (RED – Visible spectrum, 665 nm)

Band 8 (NIR – Near Infrared, 842 nm)

🌱 NDVI Feature Extraction

Vegetation health is measured using the Normalized Difference Vegetation Index (NDVI):

𝑁
𝐷
𝑉
𝐼
=
𝑁
𝐼
𝑅
−
𝑅
𝐸
𝐷
𝑁
𝐼
𝑅
+
𝑅
𝐸
𝐷
NDVI=
NIR+RED
NIR−RED
	​

Interpretation:

NDVI ∈ [-1, +1]

Higher NDVI → healthier vegetation

Lower NDVI → stressed or sparse vegetation

NDVI values are computed per pixel, masked by farm boundaries, and time-stamped for temporal analysis.

🔍 Data Acquisition Pipeline

Farm boundaries defined using GeoJSON

Sentinel-2 imagery retrieved via Sentinelsat (Copernicus API)

Relevant spectral bands downloaded

Data stored locally for preprocessing

🧪 Preprocessing Pipeline

Crop satellite images to farm boundaries

Extract RED and NIR bands

Compute NDVI for each timestamp

Normalize NDVI values

Stack multi-temporal NDVI images into tensors

Generate structured inputs for deep learning

Tools Used:

GDAL / Rasterio – Read and process geospatial raster data

OpenCV – Image resizing and cropping

NumPy – Numerical operations

Matplotlib – Data visualization

🤖 Model Architecture

A custom CNN regression model built using TensorFlow:

Architecture Details:

Input Shape: (1, 32, 10, 13) NDVI tensor

Layers:

3 Convolutional blocks

Convolution + ReLU

Batch Normalization

Dropout (regularization)

Flatten layer

Fully Connected (Dense) layers

Output: Continuous crop yield value

Training Configuration:

Loss Function: Mean Squared Error (L2 Loss)

𝑀
𝑆
𝐸
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
MSE=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−
y
^
	​

i
	​

)
2

Optimizer: Adam

Regularization: Dropout + L2 weight decay

🧩 System Architecture
flowchart TD
    A[Farm Coordinates / GeoJSON] --> B[Sentinelsat API]
    B --> C[Sentinel-2 Satellite Images]
    C --> D[Extract RED & NIR Bands]
    D --> E[NDVI Computation]
    E --> F[NDVI Time-Series Tensor]
    F --> G[CNN Model (TensorFlow)]
    G --> H[Crop Yield Prediction]

📈 Results & Performance

Processed 50+ NDVI tensors from multi-temporal Sentinel-2 imagery

Achieved high correlation (~0.82) between predicted and actual yields

Validation loss showed smooth convergence

NDVI trends aligned with known seasonal crop growth patterns

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
# Step 1: Download Sentinel-2 satellite data
python Sentinel_data_scraper.py

# Step 2: Preprocess data and compute NDVI
python Cropping_preprocessing.py

# Step 3: Train the CNN model
python Model_TF.py

# Step 4: Evaluate results and visualize predictions

🌍 Real-World Applications

Precision agriculture and smart farming

Seasonal crop yield forecasting

Government agricultural planning

Crop insurance risk assessment

Food supply chain optimization

🔮 Future Enhancements

Integrate weather and soil moisture data

Use transformer-based temporal models

Deploy as a web-based decision support system

Extend to multi-crop and multi-region prediction

📄 Summary

This project demonstrates a complete AI-powered pipeline that transforms raw satellite imagery into accurate, farm-level crop yield predictions. By automating data acquisition, NDVI feature extraction, and deep learning-based modeling, it provides a scalable and practical solution for real-world agritech and sustainability initiatives.
