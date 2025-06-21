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