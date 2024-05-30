from sklearnex import patch_sklearn
patch_sklearn()

from pickle import dump
from sklearn.ensemble import StackingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from skopt.space import Real, Integer, Categorical

est = [
    ( 'knn', 
        Pipeline(
            [
                ( 'scaling', StandardScaler() ),
                ( 'pca', PCA(0.9) ),
                ( 'classifier', KNeighborsClassifier() )
            ]
        ) 
    ),
    ( 'dt', DecisionTreeClassifier( max_depth=8 ) ),
    ( 'lr', LogisticRegression( solver='saga', penalty='l1', max_iter=1000 ) )
]

final_est = [
    LogisticRegression( solver='lbfgs' ),
    DecisionTreeClassifier( max_depth=8 ),
    KNeighborsClassifier( n_neighbors=10 ),
    BernoulliNB()
]

stck = {
    'name': 'Stacking',
    'model': StackingClassifier( estimators=est ),
    'search_spaces': {
        'knn__classifier__n_neighbors': Integer( 35, 65 ),
        'knn__classifier__weights': Categorical( [ 'uniform', 'distance' ] ),
        'dt__criterion': Categorical( [ 'gini', 'entropy' ] ),
        'lr__C': Real( 0.005, 200, prior='log-uniform' ),
        'final_estimator': Categorical(final_est),
        '':  Categorical( [ True, False ] )
    }
}

models = [ stck ]

with open( 'stacking_models.pkl', 'wb' ) as f:
    dump( models, f )

with open('stacking_list.txt', 'w') as file:
    for item in models:
        file.write(f"{item['name']} Base\n")
        file.write(f"{item['name']} Balanced\n")