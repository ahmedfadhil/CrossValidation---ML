from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

dataset = load_digits()

X, y = dataset.data, dataset.target == 1
clf = SVC(kernel='linear', C=1)

# Model evaluation to average performance
print('Cross validation (Accuracy):',
      cross_val_score(clf, X, y, cv=5))
print('Cross validation (AUC):',
      cross_val_score(clf, X, y, cv=5, scoring='roc_auc'))
print('Cross validation (Recall):',
      cross_val_score(clf, X, y, cv=5, scoring='recall'))

# Using Grid search to find the gamma

# dataset = load_digits()

# X, y = dataset.data, dataset.target == 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}
grid_clf_acc = GridSearchCV(clf, param_grid=grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)
print('Grid best parameter (max. accuracy):', grid_clf_acc.best_params_)
print('Grid best score (accuracy):', grid_clf_acc.best_score_)

grid_clf_auc = GridSearchCV(clf, param_grid=grid_values, scoring='roc_auc')
grid_clf_auc.fit(X_train, y_train)

y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)

print('Test set AUC:', roc_auc_score(y_test, y_decision_fn_scores_acc))
print('Grid Best Parameter (max AUC):', grid_clf_auc.best_params_)
print('Grid Best Score (AUC):', grid_clf_acc.best_score_)

# To get evaluation metrics supported for model selection
from sklearn.metrics.scorer import SCORERS

print(sorted(list(SCORERS.keys())))
