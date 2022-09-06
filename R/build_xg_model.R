#################################
#                               #
#                    ######     #
#     hockeyR      ###    ###   #
#                 ###     ###   #
#     ##    ##   ###            #
#      ##  ##    ###            #
#       ###      ###   ######   #
#      ##  ##     ###    ###    #
#     ##    ##      #####       #
#                               #
#   an expected goals model     #
#################################

library(tidyverse)
library(xgboost, exclude = "slice")
`%not_in%` <- purrr::negate(`%in%`)

############ LOAD DATA ##############

# load pbp data
#   -can exclude shift change events for this

pbp_all <- hockeyR::load_pbp(2011:2022)

############# DATA CLEANING #############

fenwick <- c("SHOT","MISSED_SHOT","GOAL")

############# PENALTY SHOT 'MODEL' ##########

pens <- pbp_all |>
  filter(period_type == "SHOOTOUT" | secondary_type == "Penalty Shot") |>
  filter(event_type %in% fenwick)

ps_xg <- pens |>
  summarise(xg = mean(event_type == "GOAL")) |>
  pull(xg)

ps_xg |> saveRDS("models/xg_model_penalty_shots.rds")

############ 5v5 MODEL ############

# features:
model_feats <- c(
  "shot_distance","shot_angle",  "shot_type","rebound","rush",
  "last_event_type","time_since_last","distance_from_last",
  "cross_ice_event","empty_net","last_x","last_y",
  # era -- year grouping of event
  # target
  "goal"
  )

pbp_shots <- pbp_all |>
  # add unique event_id to join original data to predictions later
  mutate(
    event_idx = str_pad(event_idx, width = 4, side = "left", pad = 0),
    event_id = as.numeric(paste0(game_id,event_idx))
  ) |>
  # filter out shootouts
  filter(period_type != "SHOOTOUT") |>
  # remove penalty shots
  filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) |>
  # add model feature variables
  group_by(game_id) |>
  mutate(
    last_event_type = lag(event_type),
    last_event_team = lag(event_team),
    time_since_last = game_seconds - lag(game_seconds),
    last_x = lag(x),
    last_y = lag(y),
    distance_from_last = round(sqrt(((y - last_y)^2) + ((x - last_x)^2)),1),
    event_zone = case_when(
      x >= -25 & x <= 25 ~ "NZ",
      (x_fixed < -25 & event_team == home_name) |
        (x_fixed > 25 & event_team == away_name) ~ "DZ",
      (x_fixed > 25 & event_team == home_name) |
        (x_fixed < -25 & event_team == away_name) ~ "OZ"
    ),
    last_event_zone = lag(event_zone)
  ) |>
  ungroup() |>
  # filter to only unblocked shots
  filter(event_type %in% fenwick) |>
  # get rid off oddball last_events
  #   ie "EARLY_INTERMISSION_START"
  filter(last_event_type %in% c("FACEOFF","GIVEAWAY","TAKEAWAY","BLOCKED_SHOT","HIT",
                           "MISSED_SHOT","SHOT","STOP","PENALTY","GOAL")) |>
  # add more feature variables
  mutate(
    era_2011_2013 = ifelse(
      season %in% c("20102011","20112012","20122013"),
      1, 0
    ),
    era_2014_2018 = ifelse(
      season %in% c("20132014","20142015","20152016","20162017","20172018"),
      1, 0
    ),
    era_2019_2021 = ifelse(
      season %in% c("20182019","20192020","20202021"),
      1, 0
    ),
    era_2022_on = ifelse(
      as.numeric(season) > 20202021, 1, 0
    ),
    # these are only for the ST model
    event_team_skaters = ifelse(event_team == home_name, home_skaters, away_skaters),
    opponent_team_skaters = ifelse(event_team == home_name, away_skaters, home_skaters),
    total_skaters_on = event_team_skaters + opponent_team_skaters,
    # these are in 5v5 model
    rebound = ifelse(last_event_type %in% fenwick & time_since_last <= 2, 1, 0),
    rush = ifelse(last_event_zone %in% c("NZ","DZ") & time_since_last <= 4, 1, 0),
    # didn't end up adding anything to model
    #set_play = ifelse(
    #  last_event_type == "FACEOFF" & time_since_last <= 4 & last_event_zone == "OZ", 1, 0
    #  ),
    cross_ice_event = ifelse(
      # indicates goalie had to move from one post to the other
      last_event_zone == "OZ" &
        ((lag(y) >  3 & y < -3) | (lag(y) < -3 & y > 3)) &
        # need some sort of time frame here to indicate shot was quick after goalie had to move
        time_since_last <= 2, 1, 0
    ),
    empty_net = ifelse(is.na(empty_net) | empty_net == FALSE, FALSE, TRUE),
    shot_type = secondary_type,
    goal = ifelse(event_type == "GOAL", 1, 0)
  ) |>
  select(season, game_id, event_id, starts_with("era"), all_of(model_feats)) |>
  # one-hot encode some categorical vars
  mutate(type_value = 1, last_value = 1) |>
  pivot_wider(names_from = shot_type, values_from = type_value, values_fill = 0) |>
  pivot_wider(
    names_from = last_event_type, values_from = last_value, values_fill = 0, names_prefix = "last_"
    ) |>
  janitor::clean_names() |>
  select(-na)

nrow(pbp_shots)
pbp_shots <- na.omit(pbp_shots)
nrow(pbp_shots)
# removed 3059 observations out of 882339 (0.3%)

# glimpse(pbp_shots)

# splitting data into testing and training

set.seed(37) #  yanni gogo gang

# keep same games in same groups to avoid leakage through last_event vars
split_shots <- rsample::group_initial_split(pbp_shots, group = game_id, prop = .8)

# training data
train_shots <- rsample::training(split_shots)

# group folds by game for CV
folds <- splitTools::create_folds(
  y = train_shots$game_id,
  k = 5,
  type = "grouped",
  invert = TRUE
)

train_shots <- train_shots |>
  select(-event_id, -season, -game_id)

test_shots <- rsample::testing(split_shots) |>
  select(-event_id, -season, -game_id)

# originally held out last two seasons
# switched to ramdom holdout with grouping as shown above
#train_shots <- pbp_shots |>
#  filter(season %not_in% c("20202021","20212022")) |>
#  select(-event_id, -season)

#test_shots <- pbp_shots |>
#  filter(season %in% c("20202021","20212022")) |>
#  select(-event_id, -season)

train_set <- train_shots |>
  select(-goal) |>
  data.matrix()

train_labels <- train_shots |>
  select(goal) |>
  data.matrix()

test_set <- test_shots |>
  select(-goal) |>
  data.matrix()

test_labels <- test_shots |>
  select(goal) |>
  data.matrix()

# xgboost format
dtrain <- xgb.DMatrix(data = train_set, label = train_labels)

dtest <- xgb.DMatrix(data = test_set, label = test_labels)

############ HYPERPARAMETER TUNING ############

# this took a while, wouldn't recommend

grid_search <- expand.grid(
  objective = "binary:logistic",
  eval_metric = "logloss",
  # round 1: max_depth = 3:10,
  # round 2: max_depth = 4:5,
  max_depth = 4,
  # round 1: eta = seq(.02,.22,.04),
  # round 2: eta = runif(20, .06, .08),
  # round 3: eta = c(.01,.035,.06),
  eta = .06,
  # round 1: gamma = seq(0,2,.5),
  # round 2: gamma = runif(20, .5, 1),
  # round 3: gamma = c(.8,1),
  gamma = 1,
  # round 1: subsample = seq(0,1,.2),
  # round 2: subsample = runif(20,.6,1),
  subsample = .8,
  # round 1: colsample_bytree = seq(0,1,.2),
  # round 2: colsample_bytree = runif(20,0,.2),
  # round 3: colsample_bytree = runif(5,.2,.9),
  colsample_bytree = .8,
  # round 1: min_child_weight = seq(0,5,1)
  # round 2: min_child_weight = 0:2,
  # round 3: min_child_weight = 0:2
  min_child_weight = 1:10
)

# for random grid search

#grid_search <- sample_n(grid_search, 100)

param_tune <- function(param){

  tictoc::tic()
  cv_model <- xgb.cv(
    data = dtrain,
    params = as.list(param),
    nfold = 5,
    nrounds = 1000,
    verbose = F,
    early_stopping_rounds = 30
  )

  cv_results <- tibble(
    logloss = min(cv_model$evaluation_log$test_logloss_mean),
    nrounds = cv_model$best_iteration,
    max_depth = param$max_depth,
    eta = param$eta,
    gamma = param$gamma,
    subsample = param$subsample,
    colsample_bytree = param$colsample_bytree,
    min_child_weight = param$min_child_weight,
    cv_rounds = cv_model$best_iteration
  )
  tictoc::toc()

  return(cv_results)
}

cv_results_5v5 <- purrr::map_dfr(
  .x = 1:nrow(grid_search),
  ~param_tune(grid_search[.x,])
)

cv_results_5v5 <- arrange(cv_results_5v5, logloss)

#cv_results_5v5 |> saveRDS("hockeyR_xg_cv_results_5v5_rd4.rds")

# examine parameter effects
cv_results_5v5 |>
  filter(nrounds > 1) |>
  pivot_longer(cols = max_depth:min_child_weight, names_to = "parameter", values_to = "value") |>
  ggplot(aes(value, logloss, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, nrow = 2, scales = "free_x")
#ggsave("figures/hockeyR_xg_cv_results_5v5_params_plot3.png", width=6,height=4)

############# CREATE REAL MODEL WITH TUNED PARAMS ################

# these were picked from the CV results
params_final <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eval_metric = "auc",
  max_depth = 4,
  eta = .06,
  gamma = 1,
  subsample = .8,
  colsample_bytree = .8,
  min_child_weight = 10
)

# do CV one more time to see metrics, get nrounds

cv_results_5v5 <- xgb.cv(
  data = dtrain,
  params = params_final,
  folds = folds,
  nfold = 5,
  nrounds = 1500,
  verbose = TRUE,
  print_every_n = 50,
  early_stopping_rounds = 30
)

cv_results_5v5 |> saveRDS("data/cv_results_5v5_final.rds")

rounds <- cv_results_5v5$best_iteration

# view results
cv_results_5v5$evaluation_log$test_logloss_mean[rounds]
cv_results_5v5$evaluation_log$test_auc_mean[rounds]

# train final model
xg_model_5v5 <- xgb.train(
  data = dtrain,
  params = params_final,
  nrounds = rounds
)

# feature importance
importance_matrix <- xgb.importance(names(train_set), model = xg_model_5v5)

# plot feature importance
importance_matrix |>
  ggplot(aes(reorder(Feature, Gain), Gain)) +
  geom_col(fill = "#99D9D9", color = "#001628") +
  coord_flip() +
  theme_bw() +
  labs(x = NULL, y = "Importance", caption = "data from hockeyR",
       title = "hockeyR 5v5 Expected Goals model feature importance")
ggsave("figures/hockeyR_xg_5v5_feature_importance.png", width = 6, height = 4, dpi = 500)

# save model
xg_model_5v5 |> saveRDS("models/xg_model_5v5.rds")

# check model on holdout test set
preds <- predict(xg_model_5v5, dtest) |>
  as_tibble() |>
  rename(xg = value) |>
  bind_cols(test_shots)

# view test set result metrics
MLmetrics::LogLoss(preds$xg, preds$goal)
pROC::auc(preds$goal, preds$xg)

############ SPECIAL TEAMS MODEL ########################

# features:
model_feats <- c(
  "shot_distance","shot_angle",  "shot_type","rebound","rush",
  "last_event_type","time_since_last","distance_from_last",
  "cross_ice_event","empty_net","last_x","last_y",
  # new for uneven strengths / short even strengths
  "total_skaters_on","event_team_advantage",
  # era,
  # target
  "goal"
)

# exclude all the faulty strength states from bad shift data
# will exclude 6v3 like younggrens do; too small of a sample
# it will only mess things up and likely won't improve anything
st_strengths <- c("5v4","5v3","6v5","6v4","4v4","4v3","3v3",
                          "4v5","3v5","5v6","4v6","3v4")

pbp_shots <- pbp_all |>
  # add unique event_id to join original data to predictions later
  mutate(
    event_idx = str_pad(event_idx, width = 4, side = "left", pad = 0),
    event_id = as.numeric(paste0(game_id,event_idx))
  ) |>
  # remove faulty strength states
  filter(strength_state %in% st_strengths) |>
  # filter out shootouts and postseason
  filter(period < 5 & season_type == "R") |>
  # remove penalty shots
  filter(secondary_type != "Penalty Shot" | is.na(secondary_type)) |>
  # add model feature variables
  group_by(game_id) |>
  mutate(
    last_event_type = lag(event_type),
    last_event_team = lag(event_team),
    time_since_last = game_seconds - lag(game_seconds),
    last_x = lag(x),
    last_y = lag(y),
    distance_from_last = round(sqrt(((y - last_y)^2) + ((x - last_x)^2)),1),
    event_zone = case_when(
      x >= -25 & x <= 25 ~ "NZ",
      (x_fixed < -25 & event_team == home_name) |
        (x_fixed > 25 & event_team == away_name) ~ "DZ",
      (x_fixed > 25 & event_team == home_name) |
        (x_fixed < -25 & event_team == away_name) ~ "OZ"
    ),
    last_event_zone = lag(event_zone)
  ) |>
  ungroup() |>
  # filter to only unblocked shots
  filter(event_type %in% fenwick) |>
  # get rid off oddball last_events
  #   ie "EARLY_INTERMISSION_START"
  filter(last_event_type %in% c("FACEOFF","GIVEAWAY","TAKEAWAY","BLOCKED_SHOT","HIT",
                                "MISSED_SHOT","SHOT","STOP","PENALTY","GOAL")) |>
  # add more feature variables
  mutate(
    era_2011_2013 = ifelse(
      season %in% c("20102011","20112012","20122013"),
      1, 0
    ),
    # goalie pad reduction, more icings, shallower nets
    era_2014_2018 = ifelse(
      season %in% c("20132014","20142015","20152016","20162017","20172018"),
      1, 0
    ),
    # goalie pad reduction
    era_2019_2021 = ifelse(
      season %in% c("20182019","20192020","20202021"),
      1, 0
    ),
    # cross checking emphasis
    era_2022_on = ifelse(
      as.numeric(season) > 20202021, 1, 0
    ),
    # new vars for ST model
    event_team_skaters = ifelse(event_team == home_name, home_skaters, away_skaters),
    opponent_team_skaters = ifelse(event_team == home_name, away_skaters, home_skaters),
    total_skaters_on = event_team_skaters + opponent_team_skaters,
    event_team_advantage = event_team_skaters - opponent_team_skaters,
    # vars for both
    rebound = ifelse(last_event_type %in% fenwick & time_since_last <= 2, 1, 0),
    rush = ifelse(last_event_zone %in% c("NZ","DZ") & time_since_last <= 4, 1, 0),
    same_event_team = ifelse(event_team == last_event_team, 1, 0),
    # didn't help, really thought it had a shot in the ST model at least
    #set_play = ifelse(
    #  last_event_type == "FACEOFF" & time_since_last <= 4 & last_event_zone == "OZ", 1, 0
    #),
    cross_ice_event = ifelse(
      # indicates goalie had to move from one post to the other
      last_event_zone == "OZ" &
        ((lag(y) >  3 & y < -3) | (lag(y) < -3 & y > 3)) &
        # need some sort of time frame here to indicate shot was quick after goalie had to move
        time_since_last <= 2, 1, 0
    ),
    empty_net = ifelse(is.na(empty_net) | empty_net == FALSE, FALSE, TRUE),
    shot_type = secondary_type,
    goal = ifelse(event_type == "GOAL", 1, 0)
  ) |>
  select(season, game_id, event_id, starts_with("era"), all_of(model_feats)) |>
  # one-hot encode some categorical vars
  mutate(type_value = 1, last_value = 1) |>
  pivot_wider(names_from = shot_type, values_from = type_value, values_fill = 0) |>
  pivot_wider(
    names_from = last_event_type, values_from = last_value, values_fill = 0, names_prefix = "last_"
  ) |>
  janitor::clean_names() |>
  select(-na)

nrow(pbp_shots)
pbp_shots <- na.omit(pbp_shots)
nrow(pbp_shots)
# 2085 observations removed out of 287377 (0.73%)

#glimpse(pbp_shots)

# train and test splitting

set.seed(37) #  yanni gogo gang

# keep all plays in single games within single set
split_shots <- rsample::group_initial_split(pbp_shots, group = game_id, prop = .8)

train_shots <- rsample::training(split_shots)

# group folds by game for CV
folds <- splitTools::create_folds(
  y = train_shots$game_id,
  k = 5,
  type = "grouped",
  invert = TRUE
)

train_shots <- train_shots |>
  select(-event_id, -season, -game_id)

test_shots <- rsample::testing(split_shots) |>
  select(-event_id, -season, -game_id)

#train_shots <- pbp_shots |>
#  filter(season %not_in% c("20202021","20212022")) |>
#  select(-event_id, -season)

#test_shots <- pbp_shots |>
#  filter(season %in% c("20202021","20212022")) |>
#  select(-event_id, -season)

train_set <- train_shots |>
  select(-goal) |>
  data.matrix()

train_labels <- train_shots |>
  select(goal) |>
  data.matrix()

test_set <- test_shots |>
  select(-goal) |>
  data.matrix()

test_labels <- test_shots |>
  select(goal) |>
  data.matrix()

dtrain <- xgb.DMatrix(data = train_set, label = train_labels)

dtest <- xgb.DMatrix(data = test_set, label = test_labels)

############# CREATE REAL MODEL WITH TUNED PARAMS ################

# One more CV to find nrounds, get preliminary result metrics
cv_results_st <- xgb.cv(
  data = dtrain,
  params = params_final,
  nfold = 5,
  folds = folds,
  nrounds = 1500,
  verbose = TRUE,
  print_every_n = 50,
  early_stopping_rounds = 30
)

cv_results_st |> saveRDS("data/cv_results_st_final.rds")

rounds <- cv_results_st$best_iteration

# check results
cv_results_st$evaluation_log$test_logloss_mean[rounds]
cv_results_st$evaluation_log$test_auc_mean[rounds]

# train final model
xg_model_st <- xgb.train(
  data = dtrain,
  params = params_final,
  nrounds = rounds
)

# feature importance
importance_matrix <- xgb.importance(names(train_set), model = xg_model_st)

# plot feature importance
importance_matrix |>
  ggplot(aes(reorder(Feature, Gain), Gain)) +
  geom_col(fill = "#99D9D9", color = "#001628") +
  coord_flip() +
  theme_bw() +
  labs(x = NULL, y = "Importance", caption = "data from hockeyR",
       title = "hockeyR Special Teams Expected Goals model feature importance")
ggsave("figures/hockeyR_xg_st_feature_importance.png", width = 6, height = 4, dpi = 500)

xg_model_st |> saveRDS("models/xg_model_st.rds")

# check on holdout test set
preds <- predict(xg_model_st, dtest) |>
  as_tibble() |>
  rename(xg = value) |>
  bind_cols(test_shots)

# metrics for test set
MLmetrics::LogLoss(preds$xg, preds$goal)
pROC::auc(preds$goal, preds$xg)
