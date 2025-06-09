import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, r2_score, make_scorer, mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

sklearn.set_config(transform_output='pandas')
df=pd.read_csv("penguins_cleaned.csv")
x=df.drop(["species","island"], axis=1)
y=df["species"]

num=["bill_length_mm","bill_depth_mm",
     "flipper_length_mm","body_mass_g"]
cat=["sex"]
one_hot=OneHotEncoder(sparse_output=False,
                      handle_unknown="infrequent_if_exist")
preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),num),
        ("cat",one_hot,cat)
    ]
)
le=LabelEncoder()
y_encoded=le.fit_transform(df["species"])

pipe=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier", RandomForestClassifier(random_state=44,
                                          max_depth=4))
])

x_train,x_test,y_train,y_test=train_test_split(
    x,y_encoded,
    train_size=0.8,
    shuffle=True,
    random_state=44,
    stratify=y_encoded
)

param_grid={
    "classifier__n_estimators":[100,200],
    "classifier__max_depth":[None,10,20],
    "classifier__min_samples_split":[2,5]
}

grid_search=GridSearchCV(
    pipe, param_grid, cv=KFold(n_splits=5,
                                shuffle=True,
                                  random_state=42),
                                    scoring="accuracy")
grid_search.fit(x_train,y_train)

y_train_pred=grid_search.predict(x_train)
y_test_pred=grid_search.predict(x_test)

