from daily_projection import DailyProjection
import datetime as dt

#Script that takes the DailyProjection class and creates projections for the day
dp_args = {
    'ev_path':'ev_game_stats',
    'pp_path':'pp_game_stats',
    'pk_path':'pk_game_stats',
    'stats_path':'daily_game_stats',
    'stats_file':'all_game_stats.csv',
    'player_pool_path':'player_pool',
    'feat_path':'daily_features_last30',
    'feat_file':'train_features.csv',
    'proj_path':'projections_last30',
    'date': dt.date(2021,3,5)
}

DP = DailyProjection(**dp_args)
#DP.write_stats_and_features()
DP.export_todays_projections()