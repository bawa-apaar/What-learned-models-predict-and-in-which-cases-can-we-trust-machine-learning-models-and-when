# What-learned-models-predict-and-in-which-cases-can-we-trust-machine-learning-models-and-when

## Introduction

In order to successfully build a machine learning model, there is a significant amount of time spent in data preprocessing. If we jump directly to train the model, either model will fail or do predictions with very low accuracy. It is only after preparing data, we fit our model with the train data and analyze its performance with test data or validation data. In this essay, we will discuss the challenges during Data Preprocessing and Data Modeling process. Further, we will discuss Model Interpretability in order to conclude when can we trust machine learning models and when is caution required.

## Challenges with Data Preprocessing

### Missing Values

Rarely, in real-world scenarios, there is a dataset without any missing values in it. It can be due to hesitancy by people to provide all the information in case of surveys, device error in case of IoT data, or human error while scraping data. Consequently, data with missing values can degrade the quality of the machine learning model drastically as missing data can lead to misleading results by introducing bias. In general, missing data can be categorized into three types: missing completely at random (MCAR), missing at random (MAR), missing not at random (MNAR). Usually, missing data can be of continuous feature (ex. Height) and categorical feature (ex. Gender). The missing feature can be imputed using Mean, Median, and mode values (Assumption: data is MCAR), Developing a model to predict missing values, etc depending on how data is distributed.

### Outlier Detection

Many times some of the data points from the dataset show more peculiar behavior than the rest of the data. It can be due to variability in data or device error. For example, we have a journey duration feature. Now for journey A, there are 100 values out of which 96 have the journey duration within 5 hours and the rest 4 have a duration of more than a day, then we can say that these 4 data points are outliers. Outliers can also mislead to missing values imputation. Though some machine learning models (generally Tree-based models) are immune to outliers, it is a good practice to handle them before building a model. Z-Score and IQR can be used to identify an outlier.

### Categorical Data
A categorical variable is data that can take on one of a limited, and usually fixed, number of possible values (eg: gender). Machine Learning models don’t understand categorical features directly. Therefore, we have to encode the categorical features before model training. Encoding a categorical feature without understanding its categories may degrade the ML model performance. Generally, encoding categorical data may categorize into two categories: Nominal encoding (one- hot encoding, mean encoding) which is used when we don’t need to rank categories (eg: pin-code), and Ordinal encoding (label encoding, target guided encoding) when we need to rank categories (eg: highest-education).

### Imbalanced Data

If there is an uneven distribution of the target class in a dataset, we call it an imbalanced dataset. For example, In Kaggle’s credit card Fraud detection dataset, the number of frauds (492) are way smaller than no frauds (284315). If we create a classification model without handling imbalanced data, our model will be biased towards the “no frauds” target class. As a result, even though we can attain high accuracy on the test dataset, our model will be considered unreliable. Therefore, handling such data becomes imperative. Smote is a common technique used to oversample minority class. As described by Dr. Bartosz Krawczyk, we can handle it by using three main approaches: Data-level (oversampling, undersampling), Algorithm-level (cost-sensitive learning), and Hybrid methods.

### Feature Selection

Not all the features are relevant for predicting the output variable. It is possible that some features have constant values or some features have too many missing values or some features are heavily correlated to each other or some features have minimum impact on the target variable. Having too many features not only makes training extremely slow, but it can also make it much harder to find a good solution. This problem is often referred to as the curse of dimensionality (Hands On Machine Learning with Scikit Learn Keras and Tensorflow, Aurelien Geron). Therefore, Jason Brownlee describes feature selection methods (PCA, Correlation, Statistical Methods) necessary to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.

## Challenges with Data Modelling

### Model Selection

Of many machine learning models, some are more restrictive (eg: Linear Regression) while others are more flexible (eg: Bagging, Boosting, SVM, neural networks). Since more restrictive models have limited applications, the question that arises is why can’t we always use more flexible models. As described in ISL, if inference is our objective, then restrictive models are much more interpretable. For instance, the linear model may be a good choice since it will be quite easy to understand the relationship between predictor and predicted variables. In contrast, when prediction is the only interest (not concerned with interpretability), then more flexible models are generally used. However, sometimes less flexible methods can lead to better predictions.

<img width="700" alt="Screenshot 2022-07-11 at 10 07 40 PM" src="https://user-images.githubusercontent.com/36688436/178314236-a34ef58e-7643-4f07-9003-6fff18052008.png">

### Performance Evaluation

At times during the training phase, our model follows errors or noise too closely along with patterns. As a result, it performs exceptionally well in predicting training data. However, such models have high variance which means it performs poorly on test data (Overfitting). On the other hand, sometimes the model is not capable enough to understand the information within training data (eg: decision tree with depth 1). Such model has high bias (underfitting). Generally, complex models are prone to “overfitting” and restrictive models may lead to “underfitting”. We can deduce whether our model has been overfitted or underfitted using different evaluation metrics like RMSE and R2 for regression model and accuracy, precision, and recall for classification problems. However, evaluation can sometimes mislead us if not properly understood. For example, a model with an imbalanced data may have high accuracy but poor recall. Furthermore, recall is important when false negatives weigh more and precision is important when false positives weigh more.
Source: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani: “An Introduction to Statistical Learning”, Springer

### Interpretation of Machine Learning Models

A common problem with most of the machine learning models is that they behave like a black box, that is, even our ML model has made an accurate prediction, the model can’t justify its predicted value. As described by Jrgen Bajorath, understanding model decisions is generally relevant for assessing the consistency of predictions and detecting potential sources of model bias. SHapley Additive exPlanations (SHAP) methodology enables the identification and prioritization of features that determine compound classification and activity prediction using any ML model.

### Conclusion

The Machine Learning model is one big block that consists of several small blocks. Some blocks we need to include and some we need to skip in order to build a reliable ML model. Moreover, understanding these blocks are paramount in order to have a clear vision of what to do. In this essay, I tried my best to explain some of those blocks concisely. If we consider the above points, we can indeed step in the right direction in order to understand the both data and model.
 
## References
• Sang Kyu Kwak and Jong Hae Kim: “Statistical data preparation: management of missing values and outliers”, Korean Journal of Anesthesiology, 2017. https://ekja.org/journal/view.php? doi=10.4097/kjae.2017.70.4.407

• SPSS White Paper: “Missing data: the hidden problem”. https://www.bauer.uh.edu/jhess/ documents/2.pdf

• Martin Bland: “An Introduction to Medical Statistics, Fourth Edition”, Oxford University Press, 2015

• Zhibin Li, Jian Zhang, Yongshun Gong, Yazhou Yao, and Qiang Wu: “Field-wise Learning for Multi-field Categorical Data”, 2020

• Bartosz Krawczyk: “Learning from Imbalanced Data Sets”, Springer, 2016. https:// link.springer.com/article/10.1007/s13748-016-0094-0#auth-Bartosz-Krawczyk

• Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P.: Smote: synthetic minority over- sampling technique. J. Artif. Intell. Res., 2002

• Zhou, Z.-H., Liu, X.-Y.: On multi-class cost-sensitive learning. Comput. Intell., 2010

• Benyamin Ghojogh, Maria N. Samad, Sayema Asif Mashhadi, Tania Kapoor, Wahab Ali, Fakhri Karray, Mark Crowley: “Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review”, 2019. https://arxiv.org/pdf/1905.02845v1.pdf

• Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani: “An Introduction to Statistical Learning”, Springer, 2nd ed. 2021

• Raquel Rodríguez-Pérez and Jürgen Bajorath: “Interpretation of machine learning models using shapley values: application to compound potency and multi-target activity predictions”, Journal of Computer-Aided Molecular Design, 2020. https://link.springer.com/article/10.1007/ s10822-020-00314-0
