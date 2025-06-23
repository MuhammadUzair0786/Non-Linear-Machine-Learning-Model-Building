# Decision Tree: Classification & Regression

This notebook demonstrates how to use Decision Tree algorithms for both classification and regression tasks using Python and scikit-learn.

## Decision Tree Classification
- Used to classify data into categories (e.g., Purchased: Yes/No).
- Visualizes decision boundaries to show how the model splits the feature space.
- No need for feature scaling.
- Can handle both numerical and categorical data.

**Steps:**
- Load and preprocess the dataset.
- Visualize data distribution with scatter plots.
- Split data into train and test sets.
- Train a `DecisionTreeClassifier`.
- Evaluate model performance using accuracy score.
- Visualize the decision boundary and tree structure.
- Make predictions for new data points.

## Decision Tree Regression
- Used to predict continuous values (e.g., Ice Cream Sales).
- Captures non-linear relationships in the data.
- No need for feature scaling.

**Steps:**
- Load and explore the regression dataset.
- Visualize the relationship between features and target.
- Split data into train and test sets.
- Train a `DecisionTreeRegressor`.
- Evaluate model performance using R² score.
- Visualize the tree structure.
- Predict target values for new inputs.

## Notes
- Decision Trees are easy to interpret and visualize.
- They can overfit, so consider tuning parameters like `max_depth` for better generalization.
- No scaling or normalization is required for Decision Trees.

---

# Decision Tree Pruning: Pre-Pruning & Post-Pruning

This notebook demonstrates how to control overfitting in Decision Tree models using **pre-pruning** and **post-pruning** techniques.

## What is Pruning?
Pruning is the process of reducing the size of a decision tree to prevent overfitting and improve generalization on unseen data.

## Pre-Pruning (Early Stopping)
- **Definition:**  
  Pre-pruning stops the tree from growing once a certain condition is met (e.g., maximum depth, minimum samples per leaf).
- **How to Apply:**  
  Set parameters like `max_depth`, `min_samples_split`, or `min_samples_leaf` when creating the DecisionTree model.
- **Benefit:**  
  Prevents the tree from becoming too complex and overfitting the training data.

**Example:**  
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
```

## Post-Pruning (Cost Complexity Pruning)
- **Definition:**  
  Post-pruning allows the tree to grow fully and then prunes back branches that have little importance, usually based on validation performance.
- **How to Apply:**  
  You can use a loop to train trees with different `max_depth` values and select the best one based on test/validation score, or use `ccp_alpha` parameter for cost complexity pruning.
- **Benefit:**  
  Finds the optimal tree size after seeing the full data, often leading to better generalization.

**Example:**  
```python
for i in range(1, 20):
    model = DecisionTreeClassifier(max_depth=i, random_state=42)
    model.fit(x_train, y_train)
    print(f"Max Depth: {i}, Train Score: {model.score(x_train, y_train):.2f}, Test Score: {model.score(x_test, y_test):.2f}")
```

## Notes
- Pruning helps avoid overfitting and improves model performance on new data.
- Always compare train and test scores to check for overfitting or underfitting.
- Visualize the effect of pruning parameters to select the best model.

---

# K-Nearest Neighbour (KNN) Classification

This notebook demonstrates how to use the K-Nearest Neighbour (KNN) algorithm for classification tasks using Python and scikit-learn.

## What is KNN?
- KNN is a simple, non-parametric algorithm used for classification and regression.
- It predicts the class of a data point based on the majority class among its 'k' nearest neighbors in the feature space.

## Steps Covered

- Load and explore the dataset.
- Visualize the data distribution using scatter plots.
- Split the data into training and testing sets.
- Train KNN classifiers with different values of 'k' to find the best accuracy.
- Evaluate model performance using train and test scores.
- Make predictions for new data points.
- Encode categorical labels for visualization.
- Plot decision boundaries to visualize how KNN separates different classes.

## Notes

- KNN works well with small to medium-sized datasets.
- Feature scaling is recommended for KNN, but in this example, original features are used.
- The value of 'k' (number of neighbors) can significantly affect model performance.

--- 

# K-Nearest Neighbour (KNN) Regression: Medical Checkup Charges

This notebook demonstrates how to use the K-Nearest Neighbour (KNN) algorithm for regression to predict medical checkup charges.

## What is KNN Regression?
- KNN regression predicts the target value for a data point based on the average (or weighted average) of its 'k' nearest neighbors in the feature space.
- It is a non-parametric, instance-based learning algorithm.

## Steps Covered

- Load and explore the Medical_Checkup_Charges dataset.
- Visualize data distribution and check for outliers.
- Handle missing values and encode categorical variables.
- Scale numerical features for better KNN performance.
- Split the data into training and testing sets.
- Train a KNN regressor and tune the value of 'k'.
- Optionally perform feature selection to improve results.
- Evaluate model performance using R² score on train and test sets.
- Predict charges for new data points.

## Notes

- Feature scaling (e.g., StandardScaler) is important for KNN regression.
- Try different values of 'k' to find the best performance.
- Removing outliers and irrelevant features can improve accuracy.
- KNN regression works best with clean, well-preprocessed data.

---
# Support Vector Machine (SVM) Classification

This notebook demonstrates how to use Support Vector Machine (SVM) for classification tasks using Python and scikit-learn.

## What is SVM?
- SVM is a powerful supervised learning algorithm used for classification and regression.
- It finds the optimal hyperplane that best separates different classes in the feature space.
- SVM works well for both linear and non-linear data (using different kernels).

## Steps Covered

- Load and explore the dataset.
- Visualize the data distribution using scatter plots.
- Split the data into training and testing sets.
- Train an SVM classifier (with linear or other kernels).
- Evaluate model performance using train and test accuracy.
- Make predictions for new data points.
- Visualize the decision boundary using `plot_decision_regions`.

## Notes

- Feature scaling is recommended for SVM, especially with non-linear kernels.
- You can experiment with different kernels (`linear`, `rbf`, `poly`) for better results.
- SVM is effective for high-dimensional spaces and when the number of features is greater than the number of samples.


This notebook is a practical guide for beginners to understand and apply SVM classification to real-

# Support Vector Machine (SVM) Regression

This notebook demonstrates how to use Support Vector Machine (SVM) for regression tasks using Python and scikit-learn.

## What is SVM Regression?
- SVM regression (SVR) is a supervised learning algorithm used to predict continuous values.
- It tries to fit the best line (or curve) within a margin, using different kernels (linear, polynomial, RBF) for linear and non-linear data.

## Steps Covered

- Load and explore the regression dataset.
- Visualize the relationship between features and target variable.
- Handle missing values and scale numerical features (important for SVM).
- Split the data into training and testing sets.
- Train an SVR model (e.g., with RBF kernel).
- Evaluate model performance using R² score on train and test sets.
- Visualize the regression curve along with the data points.
- Predict target values for new inputs.

## Notes

- Feature scaling (e.g., StandardScaler) is important for SVR to perform well.
- You can experiment with different kernels (`linear`, `rbf`, `poly`) for best results.
- SVR is effective for capturing both linear and non-linear relationships.


This notebook is a practical guide for beginners to understand and apply SVM regression to real-

---

# Hyperparameter Tuning: Grid Search CV & Randomized Search CV

This README explains how to use **GridSearchCV** and **RandomizedSearchCV** for hyperparameter tuning in machine learning models using scikit-learn.

---

## What is Hyperparameter Tuning?

Hyperparameters are model settings that are not learned from the data but set before training (e.g., `n_neighbors` in KNN, `C` in SVM).  
Tuning these can significantly improve model performance.

---

## Grid Search CV

- **GridSearchCV** exhaustively tries all combinations of specified hyperparameter values.
- It uses cross-validation to evaluate each combination and selects the best one based on a scoring metric.

**Example:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid = GridSearchCV(
    estimator=KNeighborsRegressor(),
    param_grid=param_grid,
    scoring='r2',        # Use 'accuracy' for classification
    cv=5,
    verbose=1
)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

---

## Randomized Search CV

- **RandomizedSearchCV** tries a fixed number of random combinations from the parameter grid.
- Useful when the parameter space is large and exhaustive search is computationally expensive.

**Example:**
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor

param_dist = {
    'n_neighbors': range(2, 20),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions=param_dist,
    n_iter=10,           # Number of random combinations to try
    scoring='r2',
    cv=5,
    verbose=1,
    random_state=42
)
random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

---

## When to Use

- **GridSearchCV:** When the parameter space is small and you want to try all combinations.
- **RandomizedSearchCV:** When the parameter space is large and you want faster results.

---

Hyperparameter tuning helps you find the best model settings for

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- mlxtend (for decision boundary plots)
- Jupyter Notebook

---

This notebook is a practical guide for beginners to understand and apply Decision Tree algorithms (classification, regression, pruning) and KNN classification to real-world