police <- readr::read_csv("/home/mach1ne/hierarchical_ts_forecaster/data/police_to_r.csv") %>%
  mutate(YearMonthDay = yearmonthday(Date)) %>%
  select(-Date)  %>%
  as_tsibble(key = c(Crime, Beat, Street, ZIP), index = YearMonthDay) %>%
  relocate(YearMonthDay)


police_gts <- police %>%
  aggregate_key((Crime * Beat * Street * ZIP), Count = sum(Count))

fit <- police_gts %>%
  filter(YearMonthDay <= yearmonthday(as.Date("2020-10-02"))) %>%
  model(base = ETS(Count)) %>%
  reconcile(
    bottom_up = bottom_up(base),
    MinT = min_trace(base, method = "mint_shrink")
  )

fc <- fit %>% forecast(h = 8)

# Crime
results_crime = fc %>%
  filter(
    is_aggregated(Beat), is_aggregated(Street), is_aggregated(ZIP)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_crime = results_crime %>% add_column(group = 'crime', .before=".model")
results = results_crime

# Beat
results_beat = fc %>%
  filter(
    is_aggregated(Crime), is_aggregated(Street), is_aggregated(ZIP)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_beat = results_beat %>% add_column(group = 'beat', .before=".model")
results = full_join(results, results_beat)

# Street
results_street = fc %>%
  filter(
    is_aggregated(Crime), is_aggregated(Beat), is_aggregated(ZIP)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_street = results_street %>% add_column(group = 'street', .before=".model")
results = full_join(results, results_street)

# ZIP
results_zip = fc %>%
  filter(
    is_aggregated(Crime), is_aggregated(Beat), is_aggregated(Street)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_zip = results_zip %>% add_column(group = 'zip', .before=".model")
results = full_join(results, results_zip)

# Total
results_tot = fc %>%
  filter(
    is_aggregated(Crime), is_aggregated(Beat), is_aggregated(Street), is_aggregated(ZIP)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_tot = results_tot %>% add_column(group = 'total', .before=".model")
results = full_join(results, results_tot)


# All
results_all = fc %>%
  accuracy(data = police_gts, measures = list(
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
    !is_aggregated(Crime), !is_aggregated(Beat), !is_aggregated(Street), !is_aggregated(ZIP)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_bot = results_bot %>% add_column(group = 'bot', .before=".model")
results = full_join(results, results_bot)


results = results %>% filter(!(.model=='base'))
results

