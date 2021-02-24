import pandas as pd
import numpy as np
import datetime as dt
import os

#Class that takes game stats and converts them into a set of features to be used for linear regression
#Takes a stat file that contains a players stats for a given day
#and takes a player pool file that contains the players playing on that day, their lines and power play lines, and vegas implied total
class Feature:
    def __init__(self,stats_file,player_pool_path):
        self.stats = {}
        self.stats['S'] = pd.read_csv(stats_file).set_index(['date','name'])
        self.stats['F'] = self.stats['S'][self.stats['S']['position'].isin(['L','C','R'])]
        self.stats['D'] = self.stats['S'][self.stats['S']['position']=='D']
        self.player_pool_path = player_pool_path
        self.stats_list = ['evTOI','evG','evA','evSH','evBkS','evixG']

    #reads in player pool file, cleans, and separates by position, returns dict of positions
    def get_player_pool(self,file_name):
        player_pool = pd.read_csv(file_name)
        player_pool['name'] = player_pool['first_name']+' '+player_pool['last_name']
        player_pool = player_pool[player_pool['injury_status']!='O']
        player_pool = player_pool[['name','team','position','reg_line','pp_line','implied_team_score']]
        player_pool = player_pool.set_index('name')

        position_pool = {}
        position_pool['F'] = player_pool[player_pool['position'].isin(['C','W'])]
        position_pool['D'] = player_pool[player_pool['position']=='D']
        position_pool['S'] = player_pool[player_pool['position'].isin(['C','W','D'])]
        position_pool['G'] = player_pool[player_pool['position']=='G']

        return position_pool

    #groups given player pool to be used for stat aggregation functions, returns dict of groups based on position
    def get_player_group(self,date):
        groups = {}
        groups['S'] = self.stats['S'].loc[:date.strftime('%y_%m_%d')][self.stats_list].dropna().groupby('name')
        groups['F'] = self.stats['F'].loc[:date.strftime('%y_%m_%d')][self.stats_list].dropna().groupby('name')
        groups['D'] = self.stats['D'].loc[:date.strftime('%y_%m_%d')][self.stats_list].dropna().groupby('name')

        return groups

    #Naive regression helper function for regressed mean function
    #If a player's games played is less than the sample size, the league average for the stat is calculated
    #and the rest of his games are simulated using that stat times a scalar to prevent small sample size distorition
    def naive_regression(self,player,df,sample_size,regression_scalar):
        temp_df = df.drop(player)
        s = pd.Series(data=np.average(temp_df,weights=temp_df['GP'],axis=0),index=temp_df.columns)
        r_stats = (df.loc[player,'GP']*df.loc[player] + regression_scalar*(sample_size-df.loc[player,'GP'])*s)/sample_size

        return r_stats

    #Returns a df of mean even strength rate stats for players in the given player pool
    #Stats are used from the last n games where n is the sample size
    #Players with fewer games played than the sample size have their stats regressed using the naive regression function
    def regressed_mean(self,group,player_pool,sample_size,regression_scalar):
        means = group.apply(lambda x: x.iloc[-sample_size:].mean())     #calculates mean stats for a player in last n games
        means = (means.drop('evTOI',axis=1).div(means['evTOI'],axis=0)*60).add_suffix('/60')
        means['GP'] = group.apply(lambda x: len(x.iloc[-sample_size:]))     #calculates games played from a player
           
        reg_means = means.join(player_pool[[]],how='inner')     #filters to only include players in player pool
        #Regresses players with GP less than sample size 
        reg_means = reg_means.index.to_series().apply(lambda x: self.naive_regression(x,means,sample_size,regression_scalar)).drop('GP',axis=1)

        return reg_means

    #Helper function that removes the given player from a line list created in get line mate stat function
    def get_line(self,team,line,name,line_list):
        if pd.notna(line):
            l = line_list.loc[team,line].copy()
            l.remove(name)
            return l
        else:
            return []

    #Helper function that gets average value of a line's stat
    def line_mate_avg(self,lm,df,stat):
        xG = []
        if not lm:
            return 0    #If line does not exist, 0 is returned
        for m in lm:
            try:
                xG.append(df.loc[m,stat])
            except:
                pass
        return pd.Series(xG,dtype='float64').mean()

    #Function that calculates the average value of a line's stat to be used in features
    def get_line_mate_stat(self,player_pool,stat_df,stat):
        #Lines, D partners, and power play lines are calculated using line numbers in player pool df
        line_mates = player_pool['F'][['team','reg_line']].copy().reset_index()     #forwards grouped by team and reg line
        lines = line_mates.groupby(['team','reg_line'])['name'].apply(list).unstack()

        d_partners = player_pool['D'][['team','reg_line']].copy().reset_index()     #defenders grouped by team and d pair
        d_pairs = d_partners.groupby(['team','reg_line'])['name'].apply(list).unstack()

        pp_line_mates = player_pool['S'][['team','pp_line']].copy().reset_index()   #all skaters grouped by pp line
        pp_lines = pp_line_mates.groupby(['team','pp_line'])['name'].apply(list).unstack()

        #Lines are calculated by get_line function and added as a column to resepective dfs
        line_mates['mates'] = line_mates.apply(lambda x: self.get_line(x['team'],x['reg_line'],x['name'],lines),axis=1)
        d_partners['mates'] = d_partners.apply(lambda x: self.get_line(x['team'],x['reg_line'],x['name'],d_pairs),axis=1)
        pp_line_mates['mates'] = pp_line_mates.apply(lambda x: self.get_line(x['team'],x['pp_line'],x['name'],pp_lines),axis=1)
        
        #Averages for line are calculated using line_mate_avg function
        line_mates[stat] = line_mates.apply(lambda x: self.line_mate_avg(x['mates'],stat_df,stat),axis=1)
        line_mates = line_mates.set_index('name')
        d_partners[stat] = d_partners.apply(lambda x: self.line_mate_avg(x['mates'],stat_df,stat),axis=1)
        d_partners = d_partners.set_index('name')
        pp_line_mates[stat] = pp_line_mates.apply(lambda x: self.line_mate_avg(x['mates'],stat_df,stat),axis=1)
        pp_line_mates = pp_line_mates.set_index('name')

        #Calculated stats are put into a df with columns for reg_line/d_pair and pp_line
        lm = pd.concat([line_mates[[stat]],d_partners[[stat]]]).add_prefix('lm')
        pplm = pp_line_mates[[stat]].add_prefix('pplm')
        lm_df = lm.join(pplm)

        return lm_df
    
    #Takes data from all other calculations and puts them into df with the actual target variable scored on that day FP/60
    #Calculates for a given  player pool, stat df (like the one returned by regressed_mean), position, and date
    def get_features(self,player_pool,stat_df,pos,date):
        lm_stats = self.get_line_mate_stat(player_pool,stat_df,'evixG/60')  #gets line mate stats
        feat = stat_df.join(lm_stats,how='inner')   #joins line_mate stats to stat_df
        feat = feat.join(player_pool[pos][['implied_team_score','position']],how='inner')   #joins imp team score and pos from player pool
        feat = feat.join(self.stats[pos].xs(date.strftime('%y_%m_%d'))['FP/60'],how='inner')    #attahces actual FP/60 on the day
        feat['date'] = date.strftime('%y_%m_%d')
        feat = feat.reset_index().set_index(['date','name','position'])
        
        return feat

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Calculates features for a range of dates and writes corresponding csvs to given file
    #Takes a path name to write to, a start and end date, and a function to calculate stats plus optional arguments that may be used by the function
    #Currently, the only function to use is regressed mean, but others can be coded and used in the future
    def write_daily_features(self,feat_path_name,start_date,end_date,stat_func,**func_args):
        #loops through dates in the range
        while start_date <= end_date:
            try:
                #Calculates player pools and groups
                player_pool = self.get_player_pool(f"{self.player_pool_path}/DFF_NHL_cheatsheet_{start_date.strftime('%Y-%m-%d')}.csv")
                player_groups = self.get_player_group(start_date)
                #Calculates stats for forwards and defenders
                player_stats = {pos:stat_func(player_groups[pos],player_pool[pos],**func_args) for pos in ['F','D']}   
                player_stats['S'] = pd.concat([player_stats['F'],player_stats['D']])
                #Calculates all features for forwards and defenders
                feat = {pos:self.get_features(player_pool,player_stats['S'],pos,start_date) for pos in ['F','D']}
                #Combines features for all skaters and writes to a csv in the given folder
                pd.concat(feat.values()).to_csv(f"{feat_path_name}/{start_date.strftime('%y_%m_%d')}.csv")
            except:
                pass
            start_date += dt.timedelta(days=1)

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Same as the write_daily_features function but instead combines all features into one file to be written with a given file name
    def write_combined_features(self,file_name,start_date,end_date,stat_func,*func_args):
        df_list = []
        while start_date <= end_date:
            try:
                player_pool = self.get_player_pool(f"{self.player_pool_path}/DFF_NHL_cheatsheet_{start_date.strftime('%Y-%m-%d')}.csv")
                player_groups = self.get_player_group(start_date)  
                player_stats = {pos:stat_func(player_groups[pos],player_pool[pos],**func_args) for pos in ['F','D']}
                player_stats['S'] = pd.concat([player_stats['F'],player_stats['D']])
                feat = {pos:self.get_features(player_pool,player_stats['S'],pos,start_date) for pos in ['F','D']}
                df_list.append(pd.concat(feat.values()))
            except:
                pass
            start_date += dt.timedelta(days=1)    
        pd.concat(df_list).to_csv(file_name)

    '''
    BE CAREFUL WITH THIS FUNCTION
    WILL OVERWRITE EXISTING FILES
    '''
    #Also creates one file for all features but uses already calculated daily feature files to avoid recalculation
    def write_concated_daily_features(self,file_name,path_name):
        file_list = os.listdir(path_name)
        df_list = []
        for file in file_list:
            df_list.append(pd.read_csv(f"{path_name}/{file}").set_index(['date','name','position']))
        pd.concat(df_list).to_csv(file_name)