import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

dataset_bundle = load_breast_cancer()
feature_matrix = pd.DataFrame(dataset_bundle.data, columns=dataset_bundle.feature_names)
target_vector = dataset_bundle.target

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    feature_matrix, target_vector, test_size=0.25, random_state=42
)

scaler_engine = StandardScaler()
X_train_scaled = scaler_engine.fit_transform(X_train_set)
X_test_scaled = scaler_engine.transform(X_test_set)

classifier_pack = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

performance_records = []

for model_label, model_algo in classifier_pack.items():

    model_algo.fit(X_train_scaled, y_train_set)
    predictions = model_algo.predict(X_test_scaled)
    prob_scores = model_algo.predict_proba(X_test_scaled)[:,1]

    acc_val = accuracy_score(y_test_set, predictions)
    auc_val = roc_auc_score(y_test_set, prob_scores)
    prec_val = precision_score(y_test_set, predictions)
    rec_val = recall_score(y_test_set, predictions)
    f1_val = f1_score(y_test_set, predictions)
    mcc_val = matthews_corrcoef(y_test_set, predictions)

    performance_records.append([
        model_label, acc_val, auc_val, prec_val, rec_val, f1_val, mcc_val
    ])

    joblib.dump(model_algo, f"model/{model_label.replace(' ','_')}.pkl")

results_table = pd.DataFrame(performance_records, columns=[
    "Model","Accuracy","AUC","Precision","Recall","F1","MCC"
])

print(results_table)

# Save sample test data for App testing
test_feature_frame = pd.DataFrame(
    X_test_scaled,
    columns=feature_matrix.columns
)

test_feature_frame["target"] = y_test_set

sample_output_path = "sample_test_data.csv"
test_feature_frame.to_csv(sample_output_path, index=False)

print(f"Sample test file saved as: {sample_output_path}")
 