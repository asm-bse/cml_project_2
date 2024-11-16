import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
df = pd.read_csv('train.csv')

#splitting data for field players and goalkeepers. processing field players first.
df_gk = df[df['position']=='GK']
df_field = df[df['position']!='GK']
df = df_field

#removing irrelevant data
columnt_to_drop = ['goalkeeping_speed', 
                   'goalkeeping_reflexes', 
                   'goalkeeping_positioning', 
                   'goalkeeping_kicking',
                   'goalkeeping_handling',
                   'goalkeeping_diving',
                   'value_eur',
                   'wage_eur',
                   'birthday_date',
                   'height_cm',
                   'weight_kg',
                   'club_name',
                   'league_name',
                   'league_level',
                   'club_jersey_number',
                   'club_loaned_from',
                   'club_joined',
                   'club_contract_valid_until',
                   'nation_jersey_number',
                   'release_clause_eur',
                   'real_face',
                   'id',
                   'short_name',
                   'overall',
                   'potential',
                   'nationality_name',
                   'body_type',
                   'international_reputation',
                   'player_tags',
                   'player_traits',
                   'work_rate'
                   ]

df = df.drop(columnt_to_drop, axis=1)


from sklearn.model_selection import train_test_split


def split_data(df):
    X = df.drop('position', axis=1)
    y = df['position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)

X_train = pd.get_dummies(X_train, prefix='position', drop_first=True)
X_test = pd.get_dummies(X_test, prefix='position', drop_first=True)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

#  GridSearchCV with F1 optimisation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    scoring='f1_weighted', 
    cv=5
)


grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

# Accuracy and F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("F1 Score:", f1)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)    


f1 = f1_score(y_test, y_pred, average='micro') 
print("F1 Score:", f1)
