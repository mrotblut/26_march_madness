library(xgboost)
library(dplyr)
library(stringr)
library(Matrix)
library(purrr)


regular <- read.csv("MRegularSeasonDetailedResults.csv")
tourney <- read.csv("MNCAATourneyDetailedResults.csv")
seeds <- read.csv("MNCAATourneySeeds.csv")
load("TeamAbility.RData")



# CLEAN SEEDS


seed_to_int <- function(seed_str) {
  if (is.na(seed_str)) return(NA_integer_)
  digits <- str_extract(seed_str, "\\d+")
  if (is.na(digits)) return(NA_integer_)
  as.integer(substr(digits, 1, 2))
}

seeds_clean <- seeds %>%
  mutate(SeedNum = vapply(Seed, seed_to_int, integer(1))) %>%
  select(Season, TeamID, SeedNum) %>%
  distinct()


# BUILD TEAM-SEASON FEATURES FROM REGULAR SEASON


win_side <- regular %>%
  transmute(
    Season = Season,
    DayNum = DayNum,
    TeamID = WTeamID,
    OppTeamID = LTeamID,
    Score = WScore,
    OppScore = LScore,
    Loc = WLoc,
    NumOT = NumOT,
    Win = 1
  )

lose_side <- regular %>%
  transmute(
    Season = Season,
    DayNum = DayNum,
    TeamID = LTeamID,
    OppTeamID = WTeamID,
    Score = LScore,
    OppScore = WScore,
    Loc = case_when(
      WLoc == "H" ~ "A",
      WLoc == "A" ~ "H",
      TRUE ~ "N"
    ),
    NumOT = NumOT,
    Win = 0
  )

games_long <- bind_rows(win_side, lose_side) %>%
  mutate(PointDiff = Score - OppScore)

team_stats <- games_long %>%
  group_by(Season, TeamID) %>%
  summarise(
    games_played = n(),
    win_pct = mean(Win, na.rm = TRUE),
    avg_score = mean(Score, na.rm = TRUE),
    avg_opp_score = mean(OppScore, na.rm = TRUE),
    avg_margin = mean(PointDiff, na.rm = TRUE),
    median_margin = median(PointDiff, na.rm = TRUE),
    ot_rate = mean(NumOT > 0, na.rm = TRUE),
    win_pct_loc_H = mean(Win[Loc == "H"], na.rm = TRUE),
    win_pct_loc_A = mean(Win[Loc == "A"], na.rm = TRUE),
    win_pct_loc_N = mean(Win[Loc == "N"], na.rm = TRUE),
    .groups = "drop"
  )

# means of empty vectors become NaN, so convert them to NA
team_stats <- team_stats %>%
  mutate(across(starts_with("win_pct_loc_"), ~ ifelse(is.nan(.x), NA_real_, .x)))

# MERGE BT + SEEDS INTO TEAM FEATURES


bt_clean <- bt %>%
  select(Season, TeamID, ability) %>%
  distinct()

team_features <- team_stats %>%
  left_join(bt_clean, by = c("Season", "TeamID")) %>%
  left_join(seeds_clean, by = c("Season", "TeamID"))

# fill missing BT and seed values
bt_median <- median(team_features[["ability"]], na.rm = TRUE)
if (is.na(bt_median)) bt_median <- 0

team_features <- team_features %>%
  mutate(
    ability := ifelse(is.na(.data[["ability"]]), bt_median, .data[["ability"]]),
    SeedNum = ifelse(is.na(SeedNum), 20, SeedNum)
  )


# BUILD TOURNAMENT MATCHUP TRAINING SET


matchups <- tourney %>%
  transmute(
    Season = Season,
    DayNum = DayNum,
    Team1ID = pmin(WTeamID, LTeamID),
    Team2ID = pmax(WTeamID, LTeamID),
    Team1Win = as.integer(WTeamID == pmin(WTeamID, LTeamID))
  )


# JOIN TEAM FEATURES FOR BOTH SIDES


t1_features <- team_features %>%
  rename_with(~ paste0("T1_", .x), .cols = everything())

t2_features <- team_features %>%
  rename_with(~ paste0("T2_", .x), .cols = everything())

matchups <- matchups %>%
  left_join(
    t1_features,
    by = c("Season" = "T1_Season", "Team1ID" = "T1_TeamID")
  ) %>%
  left_join(
    t2_features,
    by = c("Season" = "T2_Season", "Team2ID" = "T2_TeamID")
  )


# ENGINEER DIFFERENCE FEATURES


base_cols <- c(
  "games_played",
  "win_pct",
  "avg_score",
  "avg_opp_score",
  "avg_margin",
  "median_margin",
  "ot_rate",
  "SeedNum",
  "ability"
)

for (col in base_cols) {
  matchups[[paste0(col, "_diff")]] <- matchups[[paste0("T1_", col)]] - matchups[[paste0("T2_", col)]]
}

loc_cols <- c("win_pct_loc_H", "win_pct_loc_A", "win_pct_loc_N")
for (col in loc_cols) {
  t1_col <- paste0("T1_", col)
  t2_col <- paste0("T2_", col)
  diff_col <- paste0(col, "_diff")
  
  if (t1_col %in% names(matchups) && t2_col %in% names(matchups)) {
    x1 <- ifelse(is.na(matchups[[t1_col]]), 0.5, matchups[[t1_col]])
    x2 <- ifelse(is.na(matchups[[t2_col]]), 0.5, matchups[[t2_col]])
    matchups[[diff_col]] <- x1 - x2
  }
}

matchups$bt_sum <- matchups[["T1_ability"]] + matchups[["T2_ability"]]
matchups$bt_abs_diff <- abs(matchups[["ability_diff"]])


# FINAL MODEL MATRIX


exclude_cols <- c(
  "Season", "DayNum", "Team1ID", "Team2ID", "Team1Win",
  "T1_TeamID", "T2_TeamID"
)

feature_cols <- setdiff(names(matchups), exclude_cols)

X_df <- matchups %>%
  select(all_of(feature_cols)) %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.x), NA_real_, .x))) %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), 0, .x)))

y <- matchups$Team1Win
season_groups <- matchups$Season

X <- as.matrix(X_df)

cat("Training rows:", nrow(X), "\n")
cat("Feature count:", ncol(X), "\n")

# CROSS-VALIDATED XGBOOST BY SEASON


unique_seasons <- sort(unique(season_groups))
oof_preds <- rep(NA_real_, length(y))
models <- list()
fold_results <- list()

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.02,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 10,
  alpha = 1.0,
  lambda = 2.0
)

logloss_vec <- function(actual, pred, eps = 1e-15) {
  if (length(actual) == 0 || length(pred) == 0) return(NA_real_)
  if (length(actual) != length(pred)) return(NA_real_)
  
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(actual * log(pred) + (1 - actual) * log(1 - pred))
}

fold_results <- list()
models <- list()
oof_preds <- rep(NA_real_, length(y))

for (s in unique_seasons) {
  train_idx <- which(season_groups != s)
  valid_idx <- which(season_groups == s)
  
  # skip bad folds
  if (length(train_idx) == 0 || length(valid_idx) == 0) {
    cat("Skipping season", s, "- empty train or validation fold\n")
    next
  }
  
  X_train <- X[train_idx, , drop = FALSE]
  X_valid <- X[valid_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_valid <- y[valid_idx]
  
  # extra safety
  if (nrow(X_train) == 0 || nrow(X_valid) == 0 || length(y_train) == 0 || length(y_valid) == 0) {
    cat("Skipping season", s, "- empty matrix or label vector\n")
    next
  }
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dvalid <- xgb.DMatrix(data = X_valid, label = y_valid)
  
  model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 1200,
    watchlist = list(train = dtrain, valid = dvalid),
    verbose = 0,
    early_stopping_rounds = 50
  )
  
  preds <- predict(model, dvalid)
  
  # safety for bad prediction output
  if (length(preds) != length(y_valid)) {
    cat("Skipping season", s, "- prediction length mismatch\n")
    next
  }
  
  oof_preds[valid_idx] <- preds
  
  fold_loss <- logloss_vec(y_valid, preds)
  
  if (length(fold_loss) == 0 || is.na(fold_loss)) {
    cat("Skipping season", s, "- fold_loss is empty or NA\n")
    next
  }
  
  best_iter <- if (!is.null(model$best_iteration) && length(model$best_iteration) == 1) {
    model$best_iteration
  } else if (!is.null(model$niter) && length(model$niter) == 1) {
    model$niter
  } else {
    1200
  }
  
  fold_results[[as.character(s)]] <- data.frame(
    Season = as.integer(s),
    n_valid = as.integer(length(valid_idx)),
    best_iteration = as.integer(best_iter),
    logloss = as.numeric(fold_loss)
  )
  
  models[[as.character(s)]] <- model
  
  cat(
    "Validation season:", s,
    "- n_valid:", length(valid_idx),
    "- logloss:", round(fold_loss, 5),
    "- best_iter:", best_iter, "\n"
  )
}

cv_results <- bind_rows(fold_results)

overall_logloss <- logloss_vec(y[!is.na(oof_preds)], oof_preds[!is.na(oof_preds)])

cat("\nOverall CV logloss:", round(overall_logloss, 5), "\n")
print(cv_results)


# FIT FINAL MODEL ON ALL DATA


best_nrounds <- round(mean(cv_results$best_iteration))
best_nrounds <- max(best_nrounds, 50)

dall <- xgb.DMatrix(data = X, label = y)

final_model <- xgb.train(
  params = xgb_params,
  data = dall,
  nrounds = best_nrounds,
  verbose = 0
)

cat("\nFinal model fitted with", best_nrounds, "rounds.\n")


# FEATURE IMPORTANCE


importance <- xgb.importance(feature_names = colnames(X), model = final_model)
print(head(importance, 25))


# PREDICTION HELPERS FOR NEW GAMES


build_matchup_features <- function(season, team1_id, team2_id, team_features_df, feature_cols, ability) {
  row <- data.frame(
    Season = season,
    Team1ID = team1_id,
    Team2ID = team2_id
  )
  
  t1 <- team_features_df %>% rename_with(~ paste0("T1_", .x), everything())
  t2 <- team_features_df %>% rename_with(~ paste0("T2_", .x), everything())
  
  row <- row %>%
    left_join(t1, by = c("Season" = "T1_Season", "Team1ID" = "T1_TeamID")) %>%
    left_join(t2, by = c("Season" = "T2_Season", "Team2ID" = "T2_TeamID"))
  
  for (col in base_cols) {
    row[[paste0(col, "_diff")]] <- row[[paste0("T1_", col)]] - row[[paste0("T2_", col)]]
  }
  
  for (col in loc_cols) {
    t1_col <- paste0("T1_", col)
    t2_col <- paste0("T2_", col)
    diff_col <- paste0(col, "_diff")
    
    if (t1_col %in% names(row) && t2_col %in% names(row)) {
      x1 <- ifelse(is.na(row[[t1_col]]), 0.5, row[[t1_col]])
      x2 <- ifelse(is.na(row[[t2_col]]), 0.5, row[[t2_col]])
      row[[diff_col]] <- x1 - x2
    }
  }
  
  row$bt_sum <- row[["T1_ability"]] + row[["T2_ability"]]
  row$bt_abs_diff <- abs(row[["ability_diff"]])
  
  # keep only model features
  for (nm in feature_cols) {
    if (!nm %in% names(row)) row[[nm]] <- 0
  }
  
  row <- row[, feature_cols, drop = FALSE]
  row[] <- lapply(row, function(x) {
    x[is.infinite(x)] <- NA_real_
    x[is.na(x)] <- 0
    x
  })
  
  as.matrix(row)
}

predict_matchup <- function(season, team_a, team_b, model, team_features_df, feature_cols) {
  # model was trained on ordered IDs
  team1 <- min(team_a, team_b)
  team2 <- max(team_a, team_b)
  
  X_pred <- build_matchup_features(
    season = season,
    team1_id = team1,
    team2_id = team2,
    team_features_df = team_features_df,
    feature_cols = feature_cols
  )
  
  p_team1 <- predict(model, X_pred)
  
  if (team_a == team1) {
    return(p_team1)
  } else {
    return(1 - p_team1)
  }
}

### TEST

TEST_SEASON <- 2025
 
train_idx <- which(season_groups < TEST_SEASON)
test_idx  <- which(season_groups == TEST_SEASON)
 
X_train <- X[train_idx, , drop = FALSE]
X_test  <- X[test_idx, , drop = FALSE]
y_train <- y[train_idx]
y_test  <- y[test_idx]
 
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)
 
xgb_params <- list(
   objective = "binary:logistic",
   eval_metric = "logloss",
   eta = 0.02,
   max_depth = 4,
   subsample = 0.8,
   colsample_bytree = 0.8,
   min_child_weight = 10,
   alpha = 1.0,
   lambda = 2.0
)
 
model_2025_test <- xgb.train(
   params = xgb_params,
   data = dtrain,
   nrounds = 1200,
   watchlist = list(train = dtrain, test = dtest),
   verbose = 0,
   early_stopping_rounds = 50
)
 
pred_test <- predict(model_2025_test, dtest)

# Brier score
brier_score <- mean((pred_test - y_test)^2)
 
cat("2025 Tournament Brier Score:", round(brier_score, 6), "\n")



##### 2026 tourney preds

matchups_2026 <- read.csv("2026_Potential_Matchups.csv")

matchups_2026 <- matchups_2026 %>%
  mutate(
    Season = 2026,
    Team1ID = as.integer(HigherSeedID),
    Team2ID = as.integer(LowerSeedID),
    Seed1 = as.numeric(HigherSeedNum),
    Seed2 = as.numeric(LowerSeedNum)
  )

# merge team features
t1 <- team_features %>% rename_with(~paste0("T1_", .x), everything())
t2 <- team_features %>% rename_with(~paste0("T2_", .x), everything())

df <- matchups_2026 %>%
  left_join(t1, by = c("Season" = "T1_Season", "Team1ID" = "T1_TeamID")) %>%
  left_join(t2, by = c("Season" = "T2_Season", "Team2ID" = "T2_TeamID"))

# override seeds from CSV
df$T1_SeedNum <- df$Seed1
df$T2_SeedNum <- df$Seed2
df$SeedNum_diff <- df$Seed1 - df$Seed2

# create model matrix
for(col in feature_cols){
  if(!col %in% names(df)) df[[col]] <- 0
}

X_new <- df[, feature_cols]

X_new[] <- lapply(X_new, function(x){
  x[is.na(x)] <- 0
  x
})

X_new <- as.matrix(X_new)

# run model
pred <- predict(final_model, X_new)

results <- matchups_2026 %>%
  mutate(
    Predictions = pred
  ) %>% 
  select(HigherSeed,HigherSeedID,HigherSeedNum,LowerSeed,LowerSeedID,LowerSeedNum,Predictions)

# save predictions
write.csv(results, "predictions_2026.csv")
