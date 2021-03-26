import pandas as pd
import sklearn.linear_model as sk

#Class that performs either OLS or ridge regression given a file of features and a list of those features to use
class Regression:
    def __init__(self,feature_file):
        #reades in feature file and separates by position
        df = pd.read_csv(feature_file).set_index(['date','name'])
        self.features = {}
        self.features['G'] = df[df['position']=='G'].drop('position',axis=1)
        self.features['S'] = df[df['position']!='G'].drop('position',axis=1)
        self.features['F'] = df[df['position'].isin(['C','W'])].drop('position',axis=1)
        self.features['D'] = df[df['position']=='D'].drop('position',axis=1)

    #fits features to target variable FP/60 using OLS
    def ols(self,pos,feature_list):
        ols = sk.LinearRegression()
        df = self.features[pos].copy()
        ols.fit(df[feature_list],df['FP/60'])
    
        return ols
    
    #fits features to target variable FP/60 using ridge
    def ridge(self,pos,feature_list):
        ridge = sk.Ridge()
        df = self.features[pos].copy()
        ridge.fit(df[feature_list],df['FP/60'])
    
        return ridge