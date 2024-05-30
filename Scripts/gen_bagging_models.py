from sklearnex import patch_sklearn
patch_sklearn()

from pickle import dump
from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from skopt.space import Real, Integer, Categorical

lr_bagg = {
    'name': 'Logistic Bagging',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('bagging', BaggingClassifier( 
                estimator = LogisticRegression( solver='saga', max_iter=2000, penalty='l2' ),
                max_samples=0.3,
                random_state = 37
            ) )
    ]),
    'search_spaces': {
        'bagging__n_estimators': Integer( 5, 30 ),
        'bagging__estimator__C': Real(1000, 1500, prior='log-uniform')
    }
}

knn_bagg = {
    'name': 'KNN Bagging',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('bagging', BaggingClassifier( 
                estimator = KNeighborsClassifier(),
                max_samples=0.3,
                random_state = 37
            ) )
    ]),
    'search_spaces': {
        'bagging__n_estimators': Integer( 5, 30 ),
        'bagging__estimator__n_neighbors': Integer( 1, 20 ),
        'bagging__estimator__weights': Categorical( [ 'uniform', 'distance' ] )
    }
}

knn_bagg_pca = {
    'name': 'KNN Bagging PCA',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('bagging', BaggingClassifier( 
                estimator = KNeighborsClassifier(),
                max_samples=0.3,
                random_state = 37
            ) )
    ]),
    'search_spaces': {
        'bagging__n_estimators': Integer( 5, 30 ),
        'bagging__estimator__n_neighbors': Integer( 30, 70 ),
        'bagging__estimator__weights': Categorical( [ 'uniform', 'distance' ] )
    }
}

dt_bagg = {
    'name': 'Descicion Tree Bagging',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('bagging', BaggingClassifier( 
                estimator = DecisionTreeClassifier(),
                random_state = 37
            ) )
    ]),
    'search_spaces': {
        'bagging__n_estimators': Integer( 5, 30 ),
        'bagging__estimator__max_depth': Integer( 4, 8 ),
        'bagging__estimator__criterion': Categorical( [ 'gini', 'entropy' ] )
    }
}

bayes_bagg_pca = {
    'name': 'Bayes Bagging PCA',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('bagging', BaggingClassifier( 
                estimator = BernoulliNB(),
                random_state = 37
            ) )
    ]),
    'search_spaces': {
        'bagging__n_estimators': Integer( 5, 30 )
    }
}

models = [ lr_bagg ]

with open( 'bagging_models.pkl', 'wb' ) as f:
    dump( models, f )

with open('bagging_list.txt', 'w') as file:
    for item in models:
        file.write(f"{item['name']} Base\n")
        file.write(f"{item['name']} Balanced\n")