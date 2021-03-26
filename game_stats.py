import pandas as pd
import datetime as dt
import re
import os

#Class that writes csv files for relevant stats of each day in given range
#Can write csv files to path that keeps all dates separate
#and can concat all dates into one csv given a file name
class GameStats:
    def __init__(self,ev_path,pp_path,pk_path,goalie_path,game_path):
        self.ev_path = ev_path
        self.pp_path = pp_path
        self.pk_path = pk_path
        self.goalie_path = goalie_path
        self.stats_list = ['Player','TOI','Goals','Total Assists','Shots','Shots Blocked','ixG']
        self.goalie_stats_list = ['Player','Team','TOI','Shots Against','Saves','Goals Against','SV%','GSAA','xG Against']

        #Series used to map team mascot names to city abbreviations used in the goalie wins df
        mas = [
            'Ducks', 'Coyotes', 'Bruins', 'Sabres', 'Hurricanes', 'Blue Jackets', 'Flames', 'Blackhawks', 'Avalanche', 
            'Stars', 'Red Wings', 'Oilers', 'Panthers', 'Kings', 'Wild', 'Canadiens', 'Devils', 'Predators',
            'Islanders', 'Rangers', 'Senators', 'Flyers', 'Penguins', 'Sharks', 'Blues', 'Lightning', 'Maple Leafs',
            'Canucks', 'Golden Knights', 'Jets', 'Capitals'
        ]
        cit = [
            'ANA', 'ARI', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL','DAL', 'DET', 'EDM', 'FLA', 'L.A', 'MIN',
            'MTL', 'N.J', 'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'S.J', 'STL', 'T.B', 'TOR', 'VAN', 'VGK', 'WPG', 'WSH'
        ]
        team_map = pd.Series(cit,mas)
        
        #dataframe with info for goalie wins and shutouts to be used in FP calculations
        self.goalie_wins = self.get_goalie_wins(game_path,team_map)

    #reads in stats from Natural Stat Trick and outputs a clean dataframe with relevant data
    def get_day_stats(self,date):
        stats_list_rename = pd.Series(['name','TOI','G','A','SH','BkS','ixG'],self.stats_list)
        goalie_stats_list_rename = pd.Series(['name','Team','TOI','SA','SV','GA','SV%','GSAA','xGA'],self.goalie_stats_list)

        #reads stats from separate paths giving game stats for even strength, power play, and penalty kill
        ev = pd.read_csv(f"{self.ev_path}/{date.strftime('%y_%m_%d')}.csv")[['Team','Position']+self.stats_list].rename(columns='ev'+stats_list_rename.drop('Player'))
        pp = pd.read_csv(f"{self.pp_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pp'+stats_list_rename.drop('Player'))
        pk = pd.read_csv(f"{self.pk_path}/{date.strftime('%y_%m_%d')}.csv")[self.stats_list].rename(columns='pk'+stats_list_rename.drop('Player'))
        gs = pd.read_csv(f"{self.goalie_path}/{date.strftime('%y_%m_%d')}.csv")[self.goalie_stats_list].rename(columns=goalie_stats_list_rename.drop('Player'))

        #combines separate dfs into one df with all relevant stats, sets index
        stats = ev.merge(pp,on='Player',how='outer')
        stats = stats.merge(pk,on='Player',how='outer')
        stats['date'] = date.strftime('%y_%m_%d')
        stats = stats.fillna(0).rename(columns={'Team':'team','Player':'name','Position':'position'}).set_index(['date','team','name','position'])
        goalie_stats = gs.rename(columns={'Team':'team','Player':'name'})
        goalie_stats['date'] = date.strftime('%y_%m_%d')
        goalie_stats['position'] = 'G'

        #calculates fantasy points with helper functions
        stats['evFP'] = stats.apply(lambda x: self.ev_fp(x['evG'],x['evA'],x['evSH'],x['evBkS']),axis=1)
        stats['ppFP'] = stats.apply(lambda x: self.pp_fp(x['ppG'],x['ppA'],x['ppSH'],x['ppBkS']),axis=1)
        stats['pkFP'] = stats.apply(lambda x: self.pk_fp(x['pkG'],x['pkA'],x['pkSH'],x['pkBkS']),axis=1)
        stats['TOI'] = stats['evTOI']+stats['ppTOI']+stats['pkTOI']
        stats['FP'] = stats['evFP']+stats['ppFP']+stats['pkFP']
        stats['FP/60'] = stats['FP']/stats['TOI']*60

        #calculates FP for goalies
        goalie_stats['xGSAA'] = goalie_stats['xGA']-goalie_stats['GA']
        goalie_stats = goalie_stats.merge(self.goalie_wins[['win','SO']],on=['date','team']).set_index(['date','team','name','position'])
        goalie_stats['FP'] = goalie_stats.apply(lambda x: self.goalie_fp(x['SV'],x['GA'],x['win'],x['SO']),axis=1)
        goalie_stats['FP/60'] = goalie_stats['FP']/goalie_stats['TOI']*60
        goalie_stats['evTOI'] = goalie_stats['TOI']

        return pd.concat([stats,goalie_stats]).fillna(0)

    #Processes game info from a string in the game_info dfs
    #returns dataframe with necessary information to calculate goalie FP
    def get_goalie_wins(self,game_path,team_map):
        hg = pd.read_csv(f"{game_path}/home.csv")['Game'].apply(lambda x: self.game_string_split(x,True))
        ag = pd.read_csv(f"{game_path}/away.csv")['Game'].apply(lambda x: self.game_string_split(x,False))
        games = pd.concat([ag,hg]).sort_index()
        games['GA'] = games['GA'].astype(int)
        games['GF'] = games['GF'].astype(int)
        games['win'] = (games['GF'] > games['GA']).astype(int)
        games['SO'] = ((games['win']==1) & (games['GA'] == 0)).astype(int)
        games['team'] = games['Team'].map(team_map)
        games['opp'] = games['opp'].map(team_map)
        games = games[['date','team','GF','opp','GA','win','SO']].set_index(['date','team'])

        return games

    #Processes string used in the game info dfs with date and score
    #returns a series with relevant info
    def game_string_split(self,s,home):
        d = re.split('(?<![A-Za-z])\s|\s(?![A-Za-z])',s.replace(',',''))
        d.remove('-')
        if home:
            l = ['date','Team','GF','opp','GA']
        else:
            l = ['date','opp','GA','Team','GF']
        s = pd.Series(d,l)
        s['date'] = s['date'].replace('-','_')[2:]
        return s


    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''

    #Generates stats df and writes it to specified location
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

    def goalie_fp(self,sv,ga,w,so):
        return .8*sv-4*ga+12*w+8*so