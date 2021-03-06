import pandas as pd
from regression import Regression

#Class uses regression class to project fantasy points
class Projection:
    #Creates Regression object with test file and separates test features by position
    def __init__(self,train_file,test_file):
        self.regression = Regression(train_file)
        df = pd.read_csv(test_file)
        self.test_features = {}
        self.test_features['S'] = df.copy()
        self.test_features['F'] = df[df['position'].isin(['C','W'])].copy()
        self.test_features['D'] = df[df['position']=='D'].copy()

    def project_ols(self,f_feature_list,d_feature_list):
        #Creates separate OLS models for forwards and defenders and uses distinct features
        f_model = self.regression.ols('F',f_feature_list)
        d_model = self.regression.ols('D',d_feature_list)
        #Predicts FP/60 using features in the test features and concats F and D to have one series
        f_proj = pd.Series(f_model.predict(self.test_features['F'][f_feature_list]),self.test_features['F']['name'])
        d_proj = pd.Series(d_model.predict(self.test_features['D'][d_feature_list]),self.test_features['D']['name'])
        pred = pd.concat([f_proj,d_proj]).rename('proj_FP/60')
        
        return pred
    
    #Same as project OLS but with ridge regression
    def project_ridge(self,f_feature_list,d_feature_list):
        f_model = self.regression.ridge('F',f_feature_list)
        d_model = self.regression.ridge('D',d_feature_list)
        f_proj = pd.Series(f_model.predict(self.test_features['F'][f_feature_list]),self.test_features['F']['name'])
        d_proj = pd.Series(d_model.predict(self.test_features['D'][d_feature_list]),self.test_features['D']['name'])
        pred = pd.concat([f_proj,d_proj]).rename('proj_FP/60')
        
        return pred

    #Takes predictions from the predict functions and calculates FP using mean time on ice
    #puts all features, whether used in regression or not, adds projections, and exports to a given file
    def export_projections(self,projections,file_name):
        df = self.test_features['S'].set_index(['date','name','position']).join(projections)
        df['proj_FP'] = df['proj_FP/60']*df['mean_TOI']/60
        df['proj_value'] = df['proj_FP']/df['salary']*1000
        df = df.sort_values('proj_FP',ascending=False)
        df.to_csv(file_name)