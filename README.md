# Canopy Vision

Technologies: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Machine Learning Algorithms (Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Naive Bayes, Support Vector Machines, Random Forests)


# Objective

The aim of this project is to build a machine learning model capable of predicting the type of forest cover for a 30m x 30m patch of land in the Roosevelt National Forest of northern Colorado. By analyzing topographic and soil-related features, the project aims to assign each land patch to one of several forest cover categories. 

This predictive system can assist forest management authorities, conservationists, and researchers in identifying vegetation distribution patterns, monitoring ecological changes, and optimizing land management strategies.

The model classifies forest cover into seven types:

Spruce/Fir

Lodgepole Pine

Ponderosa Pine

Cottonwood/Willow

Aspen

Douglas-fir

Krummholz


# Problem Statement

Accurately identifying forest cover types is a critical task for sustainable forest management, ecological research, and environmental monitoring. Traditionally, forest cover classification has relied on manual surveys and fieldwork, which are labor-intensive, time-consuming, and prone to human error. With the availability of large-scale remote sensing and environmental datasets, there is an opportunity to apply machine learning techniques to automate and improve the accuracy of cover type prediction.

The Forest Cover Type Prediction System aims to leverage environmental features such as elevation, slope, soil type, and geographic attributes to classify land areas into distinct forest cover categories. This predictive capability can assist forestry departments, ecologists, and land planners in:

Efficiently mapping vegetation over large regions.

Monitoring ecological balance and changes in forest distribution.

Supporting conservation efforts by identifying vulnerable or changing habitats.

Enabling data-driven decision-making for land use and natural resource management.

The challenge lies in handling the complex relationships among multiple environmental variables while ensuring high accuracy across all forest cover classes. Therefore, building a robust machine learning model that generalizes well to unseen data is the primary objective of this system.


# Solution

Built a machine learning system to classify forest cover types into seven categories (e.g., Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz) using structured environmental data such as elevation, slope, aspect, and soil type. Performed data preprocessing by handling imbalances, normalizing continuous features, and encoding categorical attributes. Implemented and compared multiple classification algorithms, including Logistic Regression, LDA, KNN, CART, Naive Bayes, SVM, and Random Forests, analyzing trade-offs between interpretability and predictive performance.

Achieved strong accuracy (above 80% with Random Forests) by tuning hyperparameters and validating models with cross-validation. Evaluated results using accuracy, precision, recall, F1-score, and confusion matrices to highlight performance across all cover types. Extracted feature importance insights to identify environmental factors most strongly influencing forest classification, supporting applications in forestry management, land planning, and ecological monitoring.


# Evaluation & Results

The trained models are tested on the hold-out test dataset. 

Evaluation metrics used include:

Accuracy Score: To measure the percentage of correctly predicted samples.

Confusion Matrix: To analyze misclassifications across forest cover categories.

Classification Report: Precision, recall, and F1-score for each cover type.

Results show that while baseline models like Logistic Regression provide moderate accuracy, ensemble models significantly improve prediction performance, achieving  ~ 85–90% accuracy.

Logistic Regression:   ~ 65–70% accuracy

Random Forest Classifier:   ~ 85–90% accuracy

Linear Discriminant Analysis:   ~ 60–65% accuracy

K-Neighbors Classifier:   ~ 75–80% accuracy

Decision Tree Classifier:   ~ 75–80% accuracy

Gaussian NB:   ~ 55–60% accuracy

SVM:   ~ 15–20% accuracy


# Feature Importance

Elevation emerged as the most significant predictor.

Soil type and distance to hydrology were also highly influential.


# Insights & Observations

Elevation is consistently the most significant predictor across models. For example, Spruce/Fir dominates higher elevations, while Ponderosa Pine occurs at mid-elevations.

Soil types and wilderness area features influences the presence of certain species and adds important categorical signals to differentiate forest types (e.g., Aspen thrives on specific soils).

Class imbalance poses challenges for minority forest types, making precision and recall crucial metrics beyond accuracy.

Ensemble models outperform linear models by capturing complex, non-linear relationships in the dataset.


# Future Scope

Potential improvements include:

Applying Deep Learning (Neural Networks) for complex feature interactions.

Performing Hyperparameter Optimization (GridSearchCV, RandomizedSearchCV).

Incorporating spatial/geographic data (e.g., satellite imagery).

Deploying the model as an API for forest management systems.

Using SMOTE or class-weight balancing to address class imbalance.


# Conclusion

The Forest Cover Type Prediction project demonstrates the effectiveness of machine learning in environmental applications.

Random Forest Classifier achieved the best performance (~90% accuracy).

Logistic Regression provided a baseline, while ensemble models achieved significantly higher accuracy.

Elevation, soil types, and hydrological features emerged as the most important predictors. 

This project highlights the potential for data-driven ecological management, with future applications in conservation planning, wildfire risk analysis, and biodiversity monitoring.
