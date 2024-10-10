# PREDICTING ADVERTISING CLICKS USING LOGISTIC REGRESSION ALGORITHM
### Brief Project Description
This machine learning project utilizes python data science libraries- pandas, numpy, matplotlib, seaborn and scikitlearn.
### Data
The data used in this project is dummy data from Jesse Portilla's course on Python for Data Science and Machine Learning on Udemy. 
### Project's Objective
To test and showcase an understanding of data preparation for machine learning, feature selection, fitting a logistic regression model and testing model performance
### Data Analysis
The dataset had 10 columns and 1000 rows of data. The goal was to develop a model to predict whether a user would click on an advertisement (represented as either 1 or 0). The features to be used on the model were:-
-- Daily time spent the site
-- Age
-- Area Income
-- Daily internet usage
-- Male (to mean gender, represented as either 1 or 0) 
Columns such as City, Country, and Timestamped were dropped as not so much insight would have been drawn given the size of the dataset.
Exploratory analysis of the data showed age to be evenly distributed but the mean was around 35 years. 
Time spent on the site was normally distributed when explored through a histogram. However, when a hue is added, those who click on an ad seem to spend less time on the site.
A pair plot to compare all columns shows 2 clusters formed on most plots to differentiate those who clicked on an ad from those who did not. It therefore makes sense to develop a logistic regression model that uses these features to predict through probability whether a person is likely to click on an ad or not. 
### Developing the model
I used the train_test_split module from the sklearn library to split the data into training and test sets in a 70:30 ratio. I then used the logistic regression module to fit a model to the training data before making predictions on the test data.
I then used confusion matrix and classification report modules to make an assessment of the model.
### Model Accuracy
From the report, the model made predictions with an overall precision score of 90%. 
