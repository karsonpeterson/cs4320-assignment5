import pandas as pd

result_df = pd.read_csv('MNCAATourneyCompactResults.csv')
seed_df = pd.read_csv('MNCAATourneySeeds.csv')

winning_seed = pd.merge(result_df, seed_df, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
winning_seed = winning_seed.drop(['TeamID'], axis=1)
winning_seed = winning_seed.rename(mapper={'Seed':'Seed1', 'WTeamID':'TeamID1'}, axis=1)
winning_seed['Result'] = 1

winning_seed = pd.merge(winning_seed, seed_df, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
winning_seed = winning_seed.drop(['TeamID'], axis=1)
winning_seed = winning_seed.rename(mapper={'Seed':'Seed2', 'LTeamID':'TeamID2'}, axis=1)
winning_seed = winning_seed.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

losing_seed = pd.merge(result_df, seed_df, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
losing_seed = losing_seed.drop(['TeamID'], axis=1)
losing_seed = losing_seed.rename(mapper={'Seed':'Seed1', 'LTeamID': 'TeamID1'}, axis=1)
losing_seed['Result'] = 0

losing_seed = pd.merge(losing_seed, seed_df, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
losing_seed = losing_seed.drop(['TeamID'], axis=1)
losing_seed = losing_seed.rename(mapper={'Seed':'Seed2', 'WTeamID': 'TeamID2'}, axis=1)
losing_seed = losing_seed.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

all_seed = pd.concat([winning_seed, losing_seed])
print(all_seed)
all_seed.to_csv('all_seed.csv', index=False)