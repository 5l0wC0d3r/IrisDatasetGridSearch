from striprtf.striprtf import rtf_to_text 
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


json_file_path = 'algoparams_from_ui.json.rtf'
dataset_path = "iris.csv"



file = open(json_file_path, 'r')
rtf_text = file.read()
pl_text = rtf_to_text(rtf_text)
data = json.loads(pl_text)

target_column = data['design_state_data']['target']['target']
feature_handling = data['design_state_data']['feature_handling']
feature_generation = data['design_state_data']['feature_generation']
feature_reduction_method = data['design_state_data']['feature_reduction']['feature_reduction_method']
hyperparameters = data['design_state_data']['hyperparameters']

dataset_path = "iris.csv"
df = pd.read_csv(dataset_path)

for feature, details in feature_handling.items():
    if details['is_selected']:
        if details['feature_variable_type'] == 'numerical':
            if details['feature_details']['missing_values'] == 'Impute':
                if details['feature_details']['impute_with'] == 'Average of values':
                    df[feature].fillna(df[feature].mean(), inplace=True)
                else:
                    impute_value = details['feature_details']['impute_value']
                    df[feature].fillna(impute_value, inplace=True)
                
        elif details['feature_variable_type'] == 'text':
            if details['feature_details']['text_handling'] == 'Tokenize and hash':
                df[feature] = df[feature].apply(lambda x: hash(x))
            else:
                pass

linear_interactions = data['design_state_data']['feature_generation']['linear_interactions']
linear_scalar_type = data['design_state_data']['feature_generation']['linear_scalar_type']
polynomial_interactions = data['design_state_data']['feature_generation']['polynomial_interactions']
explicit_pairwise_interactions = data['design_state_data']['feature_generation']['explicit_pairwise_interactions']

# Add linear interactions
if linear_interactions:
    for interaction in linear_interactions:
        feature_name = '_'.join(interaction)
        df[f'linear_interaction_{feature_name}'] = df[interaction[0]] * df[interaction[1]]

# Add polynomial interactions
if polynomial_interactions:
    for interaction in polynomial_interactions:
        interaction_terms = interaction.split('/')
        feature_name = '_'.join(interaction_terms)
        df[f'polynomial_interaction_{feature_name}'] = df[interaction_terms[0]] / df[interaction_terms[1]]

# Add explicit pairwise interactions
if explicit_pairwise_interactions:
    for interaction in explicit_pairwise_interactions:
        interaction_terms = interaction.split('/')
        feature_name = '_'.join(interaction_terms)
        df[f'explicit_pairwise_interaction_{feature_name}'] = df[interaction_terms[0]] / df[interaction_terms[1]]


feature_reduction_method = data['design_state_data']['feature_reduction']['feature_reduction_method']
num_of_features_to_keep = int(data['design_state_data']['feature_reduction']['num_of_features_to_keep'])
num_of_trees = int(data['design_state_data']['feature_reduction']['num_of_trees'])
depth_of_trees = int(data['design_state_data']['feature_reduction']['depth_of_trees'])

def tree_based_feature_reduction(df, target_column, num_of_features_to_keep, num_of_trees, depth_of_trees):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = RandomForestRegressor(n_estimators=num_of_trees, max_depth=depth_of_trees, random_state=0)
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Sort indices based on feature importances
    indices = np.argsort(feature_importances)[::-1]

    # Select top 'num_of_features_to_keep' features
    top_feature_indices = indices[:num_of_features_to_keep]
    selected_features = X.columns[top_feature_indices]

    # Keep only the selected features in the DataFrame
    df_reduced = df[selected_features]
    df_reduced[target_column] = y  # Adding back the target column

    return df_reduced

if feature_reduction_method == 'No Reduction':
    # No reduction needed, so keep all original features
    pass

elif feature_reduction_method == 'Corr with Target':
    # Calculate correlation with the target variable
    target_column = data['design_state_data']['target']['target']
    correlation = df.corr()[target_column].abs().sort_values(ascending=False)
    
    # Select the top 'num_of_features_to_keep' features based on correlation
    selected_features = correlation.index[:num_of_features_to_keep]
    df_reduced = df[selected_features]

elif feature_reduction_method == 'Tree-based':
    df_reduced = tree_based_feature_reduction(df, target_column, num_of_features_to_keep, num_of_trees, depth_of_trees)

elif feature_reduction_method == 'PCA':
    pca = PCA(n_components=num_of_features_to_keep)
    features = df.drop(columns=[target_column])
    pca.fit(features)
    transformed_features = pca.transform(features)
    selected_features = [f'PC{i}' for i in range(1, num_of_features_to_keep + 1)]
    df_reduced = pd.DataFrame(transformed_features, columns=selected_features)
    df_reduced.loc[:, target_column] = df[target_column]

else:
    # Invalid feature reduction method specified
    raise ValueError("Invalid feature reduction method specified in the JSON.")

prediction_type = data['design_state_data']['target']['prediction_type']
algorithm_details = data['design_state_data']['algorithms']
#picking algorithm
selected_algorithms = []
for algo_name, algo_data in algorithm_details.items():
    if algo_data['is_selected']:
        selected_algorithms.append(algo_name)

grid_search_models = {}
for algo_name in selected_algorithms:
    algo_data = algorithm_details[algo_name]
    if algo_name == 'RandomForestRegressor':
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': list(range(algo_data['min_trees'], algo_data['max_trees'] + 1)),
            'min_samples_leaf': list(range(algo_data['min_samples_per_leaf_min_value'], algo_data['min_samples_per_leaf_max_value'] + 1)),
            'max_depth' : list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'n_jobs' : [algo_data['parallelism'] + 1]
        }
    elif algo_name == 'RandomForestClassifier':
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': list(range(algo_data['min_trees'], algo_data['max_trees'] + 1)),
            'min_samples_leaf': list(range(algo_data['min_samples_per_leaf_min_value'], algo_data['min_samples_per_leaf_max_value'] + 1)),
            'max_depth' : list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'n_jobs' : [algo_data['parallelism'] + 1]
        }
    elif algo_name == 'LinearRegression':
        model = LinearRegression()
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'n_jobs': [algo_data['parallelism']+1]
        }
    elif algo_name == 'LogisticRegression':
        model = LogisticRegression()
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['none', 'l2'],
            'n_jobs' : [algo_data['parallelism'] + 1]
        }
    elif algo_name == 'RidgeRegression':
        model = Ridge()
        param_grid = {
            'alpha': [0.1, 1, 10],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter' : list(range(algo_data['min_iter'], algo_data['max_iter'] + 1))
        }
    elif algo_name == 'LassoRegression':
        model = Lasso()
        param_grid = {
            'alpha': [0.1, 1, 10],
            'max_iter' : list(range(algo_data['min_iter'], algo_data['max_iter'] + 1))

        }
    elif algo_name == 'ElasticNetRegression':
        model = ElasticNet()
        param_grid = {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.1, 0.5, 0.9],            
            'max_iter' : list(range(algo_data['min_iter'], algo_data['max_iter'] + 1))

        }
    elif algo_name == 'xg_boost':
        model = xgb.XGBRegressor()
        param_grid = {
            'booster' : ['gbtree','dart'],
            'max_depth': algo_data['max_depth_of_tree'],
            'learning_rate': algo_data['learningRate'],
            'gamma': algo_data['gamma'],
            'alpha' : algo_data['l1_regularization'],
            'lambda' : algo_data['l2_regularization'],
            'subsample': algo_data['sub_sample'],
            'min_child_weight' : algo_data['min_child_weight'],
            'colsample_bytree': algo_data['col_sample_by_tree']
        }
    elif algo_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
        param_grid = {
            'max_depth': list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'min_samples_leaf': algo_data['min_samples_per_leaf'],
            'splitter': ['best', 'random']
        }
    elif algo_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'min_samples_leaf': algo_data['min_samples_per_leaf'],
            'splitter': ['best', 'random']
        }
    elif algo_name == 'SGD':
        model = SGDRegressor()
        param_grid = {
            'alpha': algo_data['alpha_value'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'tol' : [algo_data['tolerance']],
            'l1_ratio': [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
        }
    elif algo_name == 'KNN':
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': algo_data['k_value'],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3],
            'n_jobs' : [algo_data['parallelism'] + 1]

        }
    elif algo_name == 'extra_random_trees':
        model = ExtraTreesRegressor()
        param_grid = {
            'n_estimators': algo_data['num_of_trees'],
            'max_depth': algo_data['max_depth'],
            'min_samples_leaf': algo_data['min_samples_per_leaf'],
            'n_jobs' : [algo_data['parallelism'] + 1]

        }
    elif algo_name == 'neural_network':
        model = MLPRegressor()
        param_grid = {
            'hidden_layer_sizes': algo_data['hidden_layer_sizes'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'alpha': algo_data['alpha_value'],
            'solver': [algo_data["solver"]],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    elif algo_name == 'GBTRegressor':
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators': algo_data['num_of_BoostingStages'],
            'subsample' : list(range(algo_data['min_subsample'],algo_data['max_subsample']+1)),
            'max_depth' : list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'random_state' : [algo_data['fixed_number']]
        }
    elif algo_name == 'GBTClassifier':
        model = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': algo_data['num_of_BoostingStages'],
            'learning_rate': algo_data['learningRate'] if len(algo_data['learningRate'])!= 0 else [0.1],
            'loss' : ['exponential'] if algo_data['use_exponential'] == 'true' else ['log_loss'],
            'subsample' : list(range(algo_data['min_subsample'],algo_data['max_subsample']+1)),
            'max_depth' : list(range(algo_data['min_depth'], algo_data['max_depth'] + 1)),
            'random_state' : [algo_data['fixed_number']]
        }
    else:
        continue

    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid_search_models[algo_name] = grid_search

grid_search_models_fitted = {}
for algo_name, grid_search in grid_search_models.items():
    X = df_reduced.drop(columns=[target_column])
    y = df_reduced[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if algo_data['is_selected']:
        if prediction_type == 'Regression':
            scoring_metric = 'r2'
        else:
            scoring_metric = 'f1'
        
        grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring_metric, cv=5)
        grid_search.fit(X_train, y_train)
        grid_search_models_fitted[algo_name] = grid_search

best_algorithm = None
best_params = None
best_score = -1

for algo_name, grid_search in grid_search_models_fitted.items():
    if grid_search.best_score_ > best_score:
        best_algorithm = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

print("Best Algorithm:", best_algorithm)
print("Best Parameters:", best_params)
print("Best Train Score:", best_score)

# Predict on the test set using the best model
y_pred = best_algorithm.predict(X_test)

if prediction_type == 'Regression':
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print("Test score:", r2)
else:  # Assuming prediction_type is 'Classification'
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    print("Test Score:", f1)