# SmartAgri-ML-Dash
  # Overview
SmartAgri-ML-Dash is an interactive web application built using Dash, a Python framework for creating analytical web applications. This application combines Machine Learning models with smart agriculture data to recommend the most suitable crops based on various soil and environmental parameters.

# Features
<b>Data Preprocessing</b>: The application loads and preprocesses agricultural data, ensuring it is ready for analysis and modeling.

<b>Support Vector Machines (SVM)</b>: Explore SVM models with linear, polynomial, and radial basis function (RBF) kernels to classify crops based on input features.

<b>Decision Tree Classifier</b>: Understand the decision-making process of a Decision Tree Classifier for crop recommendation.

<b>Random Forest Classifier</b>: Experience the power of Random Forest, an ensemble learning method, for precise crop predictions.

<b>Gradient Boosting Classifier</b>: Learn how Gradient Boosting enhances model accuracy in smart agriculture applications.

<b>K-Nearest Neighbors (KNN) Classifier</b>: Implement KNN to recommend crops based on the similarity of their attributes.

<b>Visualization</b>: The app offers interactive visualizations like heatmaps, scatter plots, histograms, and box plots for deeper insights.

<b>Correlation Analysis</b>: Explore correlations between different agricultural features to better understand their relationships.

<b>Crop Counts</b>: Visualize the distribution of crops in the dataset to gain insights into their representation.

<b>Pair Plot</b>: Observe the relationships between various features in the dataset through a pair plot.

<b>Crop Trends</b>: Mock trends are provided to simulate crop data if the original dataset is not found.

<b>KNN Accuracy Plot</b>: Analyze the accuracy of the KNN model at different k-values using an interactive line plot.

# Instructions
 - Clone this repository to your local machine.</br>
 - Install the required libraries by running <code>pip install -r requirements.txt.</code></br>
 - Run the app using <code>python app.py </code></br>
 - Access the application on your local server at <code> http://127.0.0.1:8050/</code></br>
 - Explore the various models, visualizations, and crop recommendations based on input data.
# Note
If the original agricultural dataset is not found, mock trends for each crop will be generated for demonstration purposes. To use your dataset, make sure it is in the correct format and update the file path in the code accordingly.

# Disclaimer
This application is intended for educational and demonstrational purposes only. The crop recommendations provided are based on simulated data trends and may not reflect real-world agricultural scenarios. For accurate agricultural decisions, consult domain experts and actual field data.
