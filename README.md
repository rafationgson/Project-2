# Summer products sold on the ecommerce website "Wish"
Dataset is from kaggle: https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish?select=summer-products-with-rating-and-performance_2020-08.csv

This is an exploratory data analysis of the summer products sold on the website called 'Wish' to see what are the correlations that lead to high sales. After the analysis, I will train a machine learning model to predict how many units an item will sell based on several features. This can be used for businesses to see what features are important in predicting how many units an item would sell online.

## Project at a glance
Code used: Python
Libraries used: Numpy, Matplotlib, Seaborn, Pandas, Scikit Learn

## Data Cleaning:
1. Added columns classifying an item as Women's fashion or Men's fashion.
2. Added columns classifying the category of an item such as a dresses, shorts, swimwear, pants, skirt, top, sportswear, onepiece, footwear, sleepwear, and accessories.
3. Added columns describing the color of the item.

## Data Exploration:
This list below contains some of the most relevant insights.
1. Average Rating had a positive correlation with units sold because of the upward trend of units sold and rating, but there was a stronger correlation with the number of ratings as compared to the actual rating with regards to number of the units sold.
2. There was no significant difference in units sold for items that were on sale vs. items that were not on sale. This can most likely be attributed to the actual demand for an item.
3. The uses of ad boosts did not significantly increase the units sold.
4. The top items sold were skirts, tops, and swimwear.
5. The top item colors sold were orange, multicolor, 'others' which consisted of various prints.

## Model building:
I split the data into 80% for training the model and 20% for testing the model, and I used regression algorithms to predict the units sold.

Models used:
1. Ridge regression
2. Lasso Regression
3. Elastic Net Regression
4. Random Forest Regression
5. Gradient Boosted Regression

The first three models were chosen because they are regularized linear regression models, and this dataset had many sparse values. Aside from linear regression, I wanted to test ensemble decision tree models, and so I chose the Random Forest and Gradient Boost models.

The measure used to evaluate the cross validation was the MAE.
Results:
| Model                       | Cross Val Score     |
|-----------------------------|---------------------|
| Ridge regression            | -2130.9549599128954 |
| Lasso Regression            | -2159.132268148492. |
| Elastic Net Regression      | -2045.5793052265071 |
| Random Forest Regression    | -1740.6853139393559 |
| Gradient Boosted Regression | -1782.4896156465913 |

The Random Forest model had the lowest MAE, and so this model was fine tuned using the GridSearchCV.
Test Predictions:
| Model                    | MAE                | Accuracy          |
|--------------------------|--------------------|-------------------|
| Random Forest Regression | 1881.9226765799258 | 0.771499570972989 |

## Findings:
Based on the Random Forest model, the rating count, price, retail price, average merchant rating, the rating count of the merchant, had a high importance in predicting how many units sold. The units sold of the dataset had a fairly wide distribution and a standard deviation of 9356.539302, and so the model reflects that, but it has a MAE that still fits the range of the units sold.
