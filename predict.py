import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv') 

columns_to_drop = [
    'goalkeeping_speed', 'goalkeeping_reflexes', 'goalkeeping_positioning',
    'goalkeeping_kicking', 'goalkeeping_handling', 'goalkeeping_diving',
    'value_eur', 'wage_eur', 'birthday_date', 'height_cm', 'weight_kg',
    'club_name', 'league_name', 'league_level', 'club_jersey_number',
    'club_loaned_from', 'club_joined', 'club_contract_valid_until',
    'nation_jersey_number', 'release_clause_eur', 'real_face',
    'short_name', 'overall', 'potential', 'nationality_name', 'body_type',
    'international_reputation', 'player_tags', 'player_traits', 'work_rate', 'skill_moves'
]
df = df.drop(columns_to_drop, axis=1)
df_test = df_test.drop([col for col in columns_to_drop if col != 'id'], axis=1)


def split_data(df):
    X = df.drop('position', axis=1)
    y = df['position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)


X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test_dataset = pd.get_dummies(df_test.drop('id', axis=1), drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test_dataset = X_test_dataset.reindex(columns=X_train.columns, fill_value=0)

scoring = {
    'f1_micro': 'f1_micro',
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro'
}


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}


grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    scoring=scoring,
    refit='f1_micro', 
    cv=5,
    verbose=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print("\nFinal Model Performance on Validation Set:")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")


classification_rep = classification_report(y_test, y_pred, zero_division=0)
print("\nClassification Report (Validation Set):")
print(classification_rep)


predictions = best_rf.predict(X_test_dataset)


df_test['position'] = predictions
final_predictions = df_test[['id', 'position']]


final_predictions.to_csv('test_predictions.csv', index=False)

print("\nPredictions on test.csv saved to 'test_predictions.csv'")
