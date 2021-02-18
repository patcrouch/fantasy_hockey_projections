import pandas as pd
import datetime as dt

#Class that writes csv files for relevant stats of each day in given range
#Can write csv files to path that keeps all dates separate
#and can concat all dates into one csv given a file name
class GameStats:
    def __init__(self,ev_path,pp_path,pk_path,start_date,end_date):
        self.ev_path = ev_path
        self.pp_path = pp_path
        self.pk_path = pk_path
        self.start_date = start_date
        self.end_date = end_date
        self.stats_list = ['Player','TOI','Goals','Total Assists','Shots','Shots Blocked','ixG']

    #reads in stats from Natural Stat Trick and outputs a clean dataframe with relevant data
    def get_day_stats(self,date):
        stats_list_rename = pd.Series(['name','TOI','G','A','SH','BkS','ixG'],self.stats_list)

        #reads stats from separate paths giving game stats for even strength, power play, and penalty kill
        ev = pd.read_csv(f"{self.ev_path}/{date.strftime('%y_%m_%d')}.csv")[['Team']+self.stats_list].rename(columns='ev'+stats_list_rename.drop('Player'))
        pp = pd.read_csv(f"{self.pp_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pp'+stats_list_rename.drop('Player'))
        pk = pd.read_csv(f"{self.pk_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pk'+stats_list_rename.drop('Player'))

        #combines separate dfs into one df with all relevant stats, sets index
        stats = ev.merge(pp,on='Player',how='outer')
        stats = stats.merge(pk,on='Player',how='outer')
        stats['date'] = date.strftime('%y_%m_%d')
        stats = stats.fillna(0).rename(columns={'Team':'team','Player':'name'}).set_index(['date','team','name'])

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
    #writes daily stats dfs into specified path name
    def write_daily_stats(self,path_name):
        date = self.start_date
        end_date = self.end_date
        delta = dt.timedelta(days=1)

        #loops through each date in range and writes file in the form yy_mm_dd in path name
        while date <= end_date:
            try:
                df = self.get_day_stats(date)
                df.to_csv(f"{path_name}/{date.strftime('%y_%m_%d')}.csv")
            except:
                pass
            date += delta

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Combines all daily dfs into one file indexed by date, team and player
    def write_combined_stats(self,file_name):
        date = self.start_date
        end_date = self.end_date
        delta = dt.timedelta(days=1)
        
        #Gets stats for each day then concats them to write to given file
        df_list = []
        while date <= end_date:
            try:
                df_list.append(self.get_day_stats(date))
            except:
                pass
            date += delta

        pd.concat(df_list).to_csv(f"{file_name}.csv")   

    #helper functions to calculate FP
    def ev_fp(self,g,a,s,b):
        return 12*g+8*a+1.6*s+1.6*b

    def pp_fp(self,g,a,s,b):
        return 12.5*g+8.5*a+1.6*s+1.6*b

    def pk_fp(self,g,a,s,b):
        return 14*g+10*a+1.6*s+1.6*b