from sklearnex import patch_sklearn
patch_sklearn()

from pickle import load, dump
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from time import time

with open( 'bagging_models.pkl', 'rb' ) as f:
    models = load( f )

with open( 'data_models.pkl', 'rb' ) as f:
    data = load( f )

X_train_base, y_train_base = data['base']
X_train_bal, y_train_bal = data['balanced']

X_train_base, y_train_base = X_train_base.copy(), y_train_base.copy()
X_train_bal, y_train_bal = X_train_bal.copy(), y_train_bal.copy()

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

if __name__ == '__main__':

    for model in models:

        name = model['name']

        print(f'Training model {name} with base dataset.')
        
        if model['search_spaces']:
            search = BayesSearchCV( estimator = model['pipe'], search_spaces = model[ 'search_spaces' ], cv = cv, n_iter=5, n_jobs=5, scoring='roc_auc', verbose=5 )
            start = time()
            search.fit( X_train_base, y_train_base )
            end = time()
            result = search
        else:
            start = time()
            model['pipe'].fit( X_train_base, y_train_base )
            end = time()
            result = model['pipe']

        with open( f'{name} Base.pkl', 'wb' ) as f:
            dump( {'model': result, 'cpu_time': end - start, 'name': name}, f )

        print(f'Training model {name} with balanced dataset.')
        
        if model['search_spaces']:
            search = BayesSearchCV( estimator = model['pipe'], search_spaces = model[ 'search_spaces' ], cv = cv, n_iter=5, n_jobs=5, scoring='roc_auc', verbose=5 )
            start = time()
            search.fit( X_train_bal, y_train_bal )
            end = time()
            result = search
        else:
            start = time()
            model['pipe'].fit( X_train_bal, y_train_bal )
            end = time()
            result = model['pipe']

        with open( f'{name} Balanced.pkl', 'wb' ) as f:
            dump( {'model': result, 'cpu_time': end - start, 'name': name}, f )

    
        


    


    