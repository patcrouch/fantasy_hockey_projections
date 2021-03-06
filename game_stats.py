import pandas as pd
import datetime as dt
import os

#Class that writes csv files for relevant stats of each day in given range
#Can write csv files to path that keeps all dates separate
#and can concat all dates into one csv given a file name
class GameStats:
    def __init__(self,ev_path,pp_path,pk_path):
        self.ev_path = ev_path
        self.pp_path = pp_path
        self.pk_path = pk_path
        self.stats_list = ['Player','TOI','Goals','Total Assists','Shots','Shots Blocked','ixG']

    #reads in stats from Natural Stat Trick and outputs a clean dataframe with relevant data
    def get_day_stats(self,date):
        stats_list_rename = pd.Series(['name','TOI','G','A','SH','BkS','ixG'],self.stats_list)

        #reads stats from separate paths giving game stats for even strength, power play, and penalty kill
        ev = pd.read_csv(f"{self.ev_path}/{date.strftime('%y_%m_%d')}.csv")[['Team','Position']+self.stats_list].rename(columns='ev'+stats_list_rename.drop('Player'))
        pp = pd.read_csv(f"{self.pp_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pp'+stats_list_rename.drop('Player'))
        pk = pd.read_csv(f"{self.pk_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pk'+stats_list_rename.drop('Player'))

        #combines separate dfs into one df with all relevant stats, sets index
        stats = ev.merge(pp,on='Player',how='outer')
        stats = stats.merge(pk,on='Player',how='outer')
        stats['date'] = date.strftime('%y_%m_%d')
        stats = stats.fillna(0).rename(columns={'Team':'team','Player':'name','Position':'position'}).set_index(['date','team','name','position'])

        #calculates fantasy points with helper functions
        stats['evFP'] = stats.apply(lambda x: self.ev_fp(x['evG'],x['evA'],x['evSH'],x['evBkS']),axis=1)
        stats['ppFP'] = stats.apply(lambda x: self.pp_fp(x['ppG'],x['ppA'],x['ppSH'],x['ppBkS']),axis=1)
        stats['pkFP'] = stats.apply(lambda x: self.pk_fp(x['pkG'],x['pkA'],x['pkSH'],x['pkBkS']),axis=1)
        stats['TOI'] = stats['evTOI']+stats['ppTOI']+stats['pkTOI']
        stats['FP'] = stats['evFP']+stats['ppFP']+stats['pkFP']
        stats['FP/60'] = stats['FP']/stats['TOI']*60

        return stats

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''

    def write_daily_stats_file(self,path_name,date):
        df = self.get_day_stats(date)
        df.to_csv(f"{path_name}/{date.strftime('%y_%m_%d')}.csv")

    #writes daily stats dfs into specified path name
    def write_daily_stats_range(self,path_name,start_date,end_date):
        #loops through each date in range and writes file in the form yy_mm_dd in path name
        while start_date <= end_date:
            try:
                self.write_daily_stats_file(path_name,start_date)
            except:
                pass
            start_date += dt.timedelta(days=1)

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Combines all daily dfs into one file indexed by date, team and player
    def write_combined_stats(self,file_name,start_date,end_date):        
        #Gets stats for each day then concats them to write to given file
        df_list = []
        while start_date <= end_date:
            try:
                df_list.append(self.get_day_stats(start_date))
            except:
                pass
            start_date += dt.timedelta(days=1)

        pd.concat(df_list).to_csv(file_name)

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Similar functionto write_combined_stats that reads in already processed daily stat csvs to avoid re-processing
    def write_concated_daily_stats(self,file_name,path_name,max_date=dt.date.today()):
        file_list = os.listdir(path_name)
        df_list = []
        for file in file_list:
            if file < max_date.strftime('%y_%m_%d'):
                df_list.append(pd.read_csv(f"{path_name}/{file}").set_index(['date','team','name','position']))
        pd.concat(df_list).to_csv(file_name)
            

    #helper functions to calculate FP
    def ev_fp(self,g,a,s,b):
        return 12*g+8*a+1.6*s+1.6*b

    def pp_fp(self,g,a,s,b):
        return 12.5*g+8.5*a+1.6*s+1.6*b

    def pk_fp(self,g,a,s,b):
        return 14*g+10*a+1.6*s+1.6*b