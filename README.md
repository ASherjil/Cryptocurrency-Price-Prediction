# Bitcoin Price Prediction

### Introduction

Predicting the price of Bitcoin is a challenging yet fascinating task due to its volatile nature. In this project, we explore various machine learning models to predict Bitcoin prices. We compare the performance of different models and identify the most effective one based on evaluation metrics.

### Dataset

The dataset used for this project was sourced from Kaggle. It contains historical data of Bitcoin prices and other relevant features. Additional features were engineered to enrich the dataset.

https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset/data

### Models

Three different models were trained and compared:

1- Long Short-Term Memory (LSTM)

2- Gated Recurrent Unit (GRU)

3- Bidirectional Gated Recurrent Unit (Bidirectional GRU)

Hyperparameter tuning was performed to identify the optimal number of units and layers for each model.

### Results

The models were evaluated using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE). The GRU model, trained on data from 2016 to 2022, was found to be the most effective. The detailed comparison of all models can be found in the notebook see file. Here are the key results:

![plot_prediction_1](https://github.com/ASherjil/Cryptocurrency-Price-Prediction/assets/92602684/6001da81-0f36-4aa4-8a74-c7c73ee25b5e)

![plot2_](https://github.com/ASherjil/Cryptocurrency-Price-Prediction/assets/92602684/5fe1154b-4f89-4bc1-8c8f-32a5013fee01)

### Usage

To run this project on your local machine, please find the Jupyter Notebook file "CW2_B820928.ipynb".

# Future Work

### Further research and potential improvements

1. We can apply another approach for predicting the Bitcoin price. One popular method is called the "CEEDMAN decomposition". Here we would split our target variable which is the closing price into seperate IMFs and train models to predict these IMFs. Aggregating the IMF predictions would give us our Bitcoin closing price. This method could potentially improve prediction accuracy. 

2. Cryptocurrency prices can be influenced by many other economic factors such as price of oil and gold. A deeper investigation into feature selection could improve our model where we would likely require data from outside the Kaggle website. 

3. Cryptocurrency prices are also in many cases effected by sentiment such as tweets, google searches/trends and online forums. Sentiment analysis could also be an important factor in improving the model's accuracy. Here we would most likely require data outside of the Kaggle website. 

### Sentiment analysis 

This involves analyzing the text of news articles, social media posts, and other textual data to determine the sentiment (positive, negative, neutral) expressed about a particular cryptocurrency. This sentiment can then be used as an input feature to our model. The intuition is that positive news might lead to price increases, while negative news might cause prices to drop. This requires a pipeline that can:

1. Continuously scrape or receive news data from various sources.

2. Preprocess the text data (tokenization, removing stop words, etc.).

3. Analyze sentiment using either pre-trained models or custom models trained on labeled financial news data.


### Online learning

Online learning models are particularly useful in domains where data is continually evolving, where we want our model to adapt to the latest trends without needing a full retraining cycle.
