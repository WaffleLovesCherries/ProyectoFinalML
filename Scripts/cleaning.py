from pandas import DataFrame, crosstab
from statsmodels.stats.outliers_influence import variance_inflation_factor
from multiprocessing import Pool
from pickle import load, dump
from numpy import fill_diagonal, triu, ones, zeros
from scipy.stats import chi2_contingency

# Cleaning methods for the DF

class garbageCollector:
    def __init__( self, X ) :
        self.values = X.values
    def get_vif( self, i ):
        return variance_inflation_factor( self.values, i )

def reduce_vif_cor( X: DataFrame, corr: DataFrame, r = 10, n_jobs = 4 ):

    fill_diagonal( corr.values, 0 )
    corr = corr.abs()
    
    print('Max corr comparison')
    while True:
        if len(X.columns) == 1: break
        # Se obtienen las mayores correlaciones para comparar sus vif
        variables = corr.where( triu( ones( corr.shape ), k=1 ).astype( bool )).stack().idxmax()
        targets = [X.columns.get_loc(col) for col in variables]
        # Se buscan eliminar las variables extremadamente correlacionadas.
        if corr.loc[variables] < 0.5: break
        vif = DataFrame()
        vif['Variable'] = variables
        bucket = garbageCollector( X )
        with Pool( processes = 2 ) as pool: vif['Value'] = pool.map( bucket.get_vif, targets )
        max = vif.max()
        if max['Value'] >= r: 
             var = max['Variable']
             val = max['Value']
             print(f'Dropped col {var} with vif {val}')
             X.drop( columns = max['Variable'], inplace = True )
        skip = vif[ vif['Value'] < r ]
        for skip_item in list( skip['Variable'] ):
            corr.drop( skip_item, axis = 0, inplace = True )
            corr.drop( skip_item, axis = 1, inplace = True )

        corr.drop( max['Variable'], axis = 0, inplace = True )
        corr.drop( max['Variable'], axis = 1, inplace = True )
        del vif

def reduce_vif( X: DataFrame, r = 10, n_jobs = 4 ):
    while True:
        if len(X.columns) == 1: break
        vif = DataFrame()
        vif['Variable'] = X.columns
        bucket = garbageCollector( X )
        with Pool( processes = n_jobs ) as pool: vif['Value'] = pool.map( bucket.get_vif, [ i for i in range(X.shape[1]) ] )
        max = vif.max()
        if max['Value'] < r: break
        var = max['Variable']
        val = max['Value']
        del vif

        print(f'Dropped col {var} with vif {val}')
        X.drop( columns = max['Variable'], inplace = True )

    return X

def chisq_matrix( X ):
    num_vars = len(X.columns)
    pvalue_matrix = zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(i+1, num_vars):
            contingency_table = crosstab(X.iloc[:, i], X.iloc[:, j])
            pvalue_matrix[i, j] = chi2_contingency(contingency_table)[1]
            pvalue_matrix[j, i] = pvalue_matrix[i, j]

    return DataFrame(pvalue_matrix, index=X.columns, columns=X.columns)

def delete_underwhelming( X, threshold = 0.2 ):
    missing_values = DataFrame( X.isnull().sum() / X.shape[0] >= threshold, columns = [f'>= {threshold}'] )
    missing_variables = list( missing_values[ missing_values[f'>= {threshold}'] ].index )
    return X.drop( columns = missing_variables ).fillna( X.mode().iloc[0] )

if __name__ == "__main__":
    with open( 'DataSafe.pkl', 'rb' ) as f:
        DataRaw: DataFrame = load( f )
        DataCat: DataFrame = load( f )

    corr = DataRaw.corr()

    corr.drop( 'isFraud', axis = 0, inplace = True )
    corr.drop( 'isFraud', axis = 1, inplace = True )
    corr.drop( 'TransactionID', axis = 0, inplace = True )
    corr.drop( 'TransactionID', axis = 1, inplace = True )

    DataClean = reduce_vif_cor( DataRaw.drop( columns = ['isFraud', 'TransactionID'] ), corr )
    DataCat = delete_underwhelming( DataCat )

    print( DataRaw.columns )

    with open( 'DataClean.pkl', 'wb' ) as f:
        dump( DataClean, f )
        dump( DataCat, f )
