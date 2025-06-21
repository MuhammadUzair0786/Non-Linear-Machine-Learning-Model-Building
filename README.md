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

# Decision Tree Pruning: Pre-Pruning & Post-Pruning

This notebook demonstrates how to control overfitting in Decision Tree models using **pre-pruning** and **post-pruning** techniques.

## What is Pruning?
Pruning is the process of reducing the size of a decision tree to prevent overfitting and improve generalization on unseen data.

---

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

---

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

## Notes
- Pruning helps avoid overfitting and improves model performance on new data.
- Always compare train and test scores to check for overfitting or underfitting.
- Visualize the effect of pruning parameters to select the best model.

---

This notebook is a practical guide for understanding and applying pre-pruning and post-pruning in decision

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



## Requirements
- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- mlxtend (for decision boundary plots)
- Jupyter Notebook

---

This notebook is a practical guide for beginners to understand and apply Decision Tree algorithms for both classification and regression problems.