# Body Weight Prediction Models

## Overview
In this project, we delve into the development and assessment of predictive models designed to estimate body weight based on height and gender. Our primary goal is to construct accurate regression models for both male and female individuals, leveraging a dataset comprising height, weight, and gender information. Prior to model building, the dataset undergoes preprocessing to standardize units, and statistical analyses are conducted to uncover underlying patterns.


### Primary Objectives

1. **Develop Gender-Specific Linear Regression Models**:
   Our foremost goal is to construct accurate regression models tailored to each gender category. By accounting for gender-specific differences, we aim to enhance the precision of body weight predictions based on height.

2. **Utilize the Weka Machine Learning Library**:
   We harness the power of the Weka machine learning library for building our regression models. Weka provides a comprehensive set of tools and algorithms, enabling efficient model development and evaluation.

3. **Create an Interactive Visualization with Java Swing and JFreeChart**:
   Visualization plays a crucial role in model interpretation and comparison. To this end, we utilize Java Swing and JFreeChart to craft an interactive visualization interface. This interface empowers users to explore and scrutinize model outcomes seamlessly.

4. **Evaluate and Compare Model Performance**:
   We employ appropriate metrics to evaluate and compare the performance of our regression models. By quantifying predictive accuracy and assessing model robustness, we gain valuable insights into their efficacy in estimating body weight.
   
## Tools and Libraries
We utilize the Weka machine learning library to construct linear regression models tailored to each gender category. By leveraging Weka's functionality, we create multiple models trained on distinct subsets of the dataset, facilitating a comparative evaluation of model performance.

## Visualization
Visualization plays a pivotal role in model interpretation and comparison within our project. We employ Java Swing and JFreeChart to develop interactive scatter plots featuring regression lines for both male and female models. This graphical user interface empowers users to seamlessly explore and contrast the models.

### Model Visualizations
![Model 1](https://github.com/BaselAbuHamed/Predict-Weight-ML/assets/107325485/e65b1a2b-b73d-42a8-9659-9a940905916c)
***
![Model 2](https://github.com/BaselAbuHamed/Predict-Weight-ML/assets/107325485/74b4a16d-fc8c-4156-b1ec-ef0509e09bc3)
***
![Model 3](https://github.com/BaselAbuHamed/Predict-Weight-ML/assets/107325485/37ef2d62-54f3-4165-a50a-7b1591602a43)
***
![Model 4](https://github.com/BaselAbuHamed/Predict-Weight-ML/assets/107325485/d91a4aad-5b53-41ab-a03b-d19effb240ce)


## Model Evaluation
To gauge the predictive efficacy of our models, we employ various performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). By calculating these metrics for each model individually, we gain insights into the accuracy of weight predictions across male and female populations.

## Usage
1. Clone the repository to your local machine.
2. Navigate to the `code` directory.
3. Execute the appropriate scripts or run the provided Jupyter notebooks for model building, visualization, and evaluation.

## Dependencies
- Weka
- JFreeChart
- Java Swing
