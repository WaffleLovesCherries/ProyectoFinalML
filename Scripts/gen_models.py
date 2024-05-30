from sklearnex import patch_sklearn
patch_sklearn()

from pickle import dump

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from skopt.space import Real, Integer, Categorical

lr = {
    'name': 'Logistic',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression( solver='saga', max_iter=1000 ) )
    ]),
    'search_spaces': {
        'classifier__penalty': Categorical( [ 'l1', 'l2' ] ),
        'classifier__C': Real(1e-4, 1e+4, prior='log-uniform')
    }
}

lr_pca = {
    'name': 'Logistic PCA',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('classifier', LogisticRegression( solver='saga', max_iter=1000 ) )
    ]),
    'search_spaces': {
        'classifier__penalty': Categorical( [ 'l1', 'l2' ] ),
        'classifier__C': Real(1e-4, 1e+4, prior='log-uniform')
    }
}

knn = {
    'name': 'KNN',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'search_spaces': {
        'classifier__n_neighbors': Integer( 1, 400 ),
        'classifier__weights': Categorical( [ 'uniform', 'distance' ] )
    }
}

knn_pca = {
    'name': 'KNN PCA',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('classifier', KNeighborsClassifier())
    ]),
    'search_spaces': {
        'classifier__n_neighbors': Integer( 1, 400 ),
        'classifier__weights': Categorical( [ 'uniform', 'distance' ] )
    }
}

dt = {
    'name': 'Descicion Tree',
    'pipe': Pipeline([
        ('classifier', DecisionTreeClassifier())
    ]),
    'search_spaces': {
        'classifier__max_depth': Integer( 4, 8 ),
        'classifier__criterion': Categorical( [ 'gini', 'entropy' ] )
    }
}

bayes = {
    'name': 'Bayesian',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', BernoulliNB())
    ]),
    'search_spaces': None
}

bayes_pca = {
    'name': 'Bayesian PCA',
    'pipe': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(0.9)),
        ('classifier', BernoulliNB())
    ]),
    'search_spaces': None
}

models = [ bayes, bayes_pca ]

with open( 'models.pkl', 'wb' ) as f:
    dump( models, f )

with open('model_list.txt', 'w') as file:
    for item in models:
        file.write(f"{item['name']} Base\n")
        file.write(f"{item['name']} Balanced\n")

