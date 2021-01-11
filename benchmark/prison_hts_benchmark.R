prison <- readr::read_csv("/home/mach1ne/hierarchical_ts_forecaster/data/prison_to_r.csv") %>%
  mutate(Quarter = yearquarter(Date)) %>%
  select(-Date)  %>%
  as_tsibble(key = c(Gender, Legal, State), index = Quarter) %>%
  relocate(Quarter)


prison_gts <- prison %>%
  aggregate_key(Gender * Legal * State, Count = sum(Count)/1e3)

fit <- prison_gts %>%
  filter(year(Quarter) <= 2014) %>%
  model(base = ETS(Count)) %>%
  reconcile(
    bottom_up = bottom_up(base),
    MinT = min_trace(base, method = "mint_shrink")
  )
fc <- fit %>% forecast(h = 8)

# State
results_state = fc %>%
  filter(
    is_aggregated(Legal), is_aggregated(Gender)
  ) %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_state = results_state %>% add_column(group = 'state', .before=".model")
results = results_state

# Gender
results_gender = fc %>%
  filter(
    is_aggregated(Legal), is_aggregated(State)
  ) %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_gender = results_gender %>% add_column(group = 'gender', .before=".model")
results = full_join(results, results_gender)

# Legal
results_legal = fc %>%
  filter(
    is_aggregated(Gender), is_aggregated(State)
  ) %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_legal = results_legal %>% add_column(group = 'legal', .before=".model")
results = full_join(results, results_legal)

# Total
results_tot = fc %>%
  filter(
    is_aggregated(Legal), is_aggregated(Gender), is_aggregated(State)
  ) %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_tot = results_tot %>% add_column(group = 'total', .before=".model")
results = full_join(results, results_tot)


# All
results_all = fc %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_all = results_all %>% add_column(group = 'all', .before=".model")
results = full_join(results, results_all)


# Bottom
results_bot = fc %>% 
  filter(
    !is_aggregated(Legal), !is_aggregated(Gender), !is_aggregated(State)
  ) %>%
  accuracy(data = prison_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse)*1e3)
results_bot = results_bot %>% add_column(group = 'bot', .before=".model")
results = full_join(results, results_bot)


results = results %>% filter(!(.model=='base'))
results





