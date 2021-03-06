from game_stats import GameStats
from feature import Feature
from projection import Projection
import datetime as dt

#Main driver class that puts all other classes in one package
#takes all necessary paths and files as arguments to be used in producing projections for the given day
class DailyProjection:
    def __init__(self,ev_path,pp_path,pk_path,stats_path,stats_file,player_pool_path,feat_path,feat_file,proj_path,date=dt.date.today()):
        self.date = date
        self.prev_date = date-dt.timedelta(days=1)
        self.ev_path = ev_path
        self.pp_path = pp_path
        self.pk_path = pk_path
        self.stats_path = stats_path
        self.stats_file = stats_file
        self.player_pool_path = player_pool_path
        self.feat_path = feat_path
        self.proj_path = proj_path
        self.feat_file = feat_file

    #Function that exports projections for the day to the given proj_path
    #Also updates stats from previous day so they can be used for future training
    def export_todays_projections(self):
        #Computes game stats from the previous day and adds them to master list of stats
        game_stats = GameStats(self.ev_path,self.pp_path,self.pk_path)
        try:
            game_stats.write_daily_stats_file(self.stats_path,self.prev_date)
        except:
            pass
        game_stats.write_concated_daily_stats(self.stats_file,self.stats_path)

        #Adds actual stats from the previous day to the feature file
        #Processes todays feature file
        feature = Feature(self.stats_file,self.player_pool_path)
        try:
            feature.write_daily_features_file(self.feat_path,self.prev_date,feature.regressed_mean,True,sample_size=30,regression_scalar=.75)
        except:
            pass
        feature.write_concated_daily_features(self.feat_file,self.feat_path,self.prev_date)
        feature.write_daily_features_file(self.feat_path,self.date,feature.regressed_mean,sample_size=30,regression_scalar=.75)

        #Calculates projected FP and exports them to the projection path
        f_list = ['evSH/60','evBkS/60','evixG/60','lmevixG/60','pplmevixG/60','implied_team_score']
        d_list = ['evSH/60','evixG/60','pplmevixG/60','implied_team_score']
        projection = Projection(self.feat_file,f"{self.feat_path}/{self.date.strftime('%y_%m_%d')}.csv")
        projection.export_projections(projection.project_ridge(f_list,d_list),f"{self.proj_path}/{self.date.strftime('%y_%m_%d')}.csv")

    #Function that updates all stats and features from a date range in case there is a change to feature or stat calculation
    def write_stats_and_features(self,start_date=dt.date(2019,10,2),end_date=dt.date.today()-dt.timedelta(days=1)):
        game_stats = GameStats(self.ev_path,self.pp_path,self.pk_path)
        game_stats.write_daily_stats_range(self.stats_path,start_date,end_date)
        game_stats.write_concated_daily_stats(self.stats_file,self.stats_path)

        feature = Feature(self.stats_file,self.player_pool_path)
        feature.write_daily_features_range(self.feat_path,start_date,end_date,feature.regressed_mean,True,sample_size=30,regression_scalar=.75)
        feature.write_concated_daily_features(self.feat_file,self.feat_path,end_date)