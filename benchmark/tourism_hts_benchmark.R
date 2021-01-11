tourism <- readr::read_csv("/home/mach1ne/hierarchical_ts_forecaster/data/tourism_to_r.csv") %>%
  mutate(YearMonth = yearmonth(Date)) %>%
  select(-Date)  %>%
  as_tsibble(key = c(State, Zone, Region, Purpose), index = YearMonth) %>%
  relocate(YearMonth)


tourism_gts <- tourism %>%
  aggregate_key((State / Zone / Region) * Purpose, Count = sum(Count))

fit <- tourism_gts %>%
  filter(YearMonth <= yearmonth(as.Date("2014-12-31"))) %>%
  model(base = ETS(Count)) %>%
  reconcile(
    bottom_up = bottom_up(base),
    MinT = min_trace(base, method = "mint_shrink")
  )
  
fc <- fit %>% forecast(h = 8)

# State
results_state = fc %>%
  filter(
    is_aggregated(Zone), is_aggregated(Region), is_aggregated(Purpose)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_state = results_state %>% add_column(group = 'state', .before=".model")
results = results_state

# Zone
results_zone = fc %>%
  filter(
    is_aggregated(State), is_aggregated(Region), is_aggregated(Purpose)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_zone = results_zone %>% add_column(group = 'zone', .before=".model")
results = full_join(results, results_zone)

# region
results_region = fc %>%
  filter(
    is_aggregated(State), is_aggregated(Zone), is_aggregated(Purpose)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_region = results_region %>% add_column(group = 'region', .before=".model")
results = full_join(results, results_region)

# purpose
results_purpose = fc %>%
  filter(
    is_aggregated(State), is_aggregated(Zone), is_aggregated(Region)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_purpose = results_purpose %>% add_column(group = 'purpose', .before=".model")
results = full_join(results, results_purpose)

# Total
results_tot = fc %>%
  filter(
    is_aggregated(State), is_aggregated(Zone), is_aggregated(Region), is_aggregated(Purpose)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_tot = results_tot %>% add_column(group = 'total', .before=".model")
results = full_join(results, results_tot)


# All
results_all = fc %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_all = results_all %>% add_column(group = 'all', .before=".model")
results = full_join(results, results_all)


# Bottom
results_bot = fc %>% 
  filter(
    !is_aggregated(State), !is_aggregated(Zone), !is_aggregated(Region), !is_aggregated(Purpose)
  ) %>%
  accuracy(data = tourism_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_bot = results_bot %>% add_column(group = 'bot', .before=".model")
results = full_join(results, results_bot)


results = results %>% filter(!(.model=='base'))
results

