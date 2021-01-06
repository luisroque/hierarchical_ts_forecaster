police <- readr::read_csv("/home/mach1ne/hierarchical_ts_forecaster/data/covid_to_r.csv") %>%
  mutate(YearMonthDay = yearmonthday(Date)) %>%
  select(-Date)  %>%
  as_tsibble(key = c(Continent, Country), index = YearMonthDay) %>%
  relocate(YearMonthDay)


police_gts <- police %>%
  aggregate_key((Continent / Country), Count = sum(Count))

fit <- police_gts %>%
  filter(YearMonthDay <= yearmonthday(as.Date("2020-11-29"))) %>%
  model(base = ETS(Count)) %>%
  reconcile(
    bottom_up = bottom_up(base),
    MinT = min_trace(base, method = "mint_shrink")
  )

fc <- fit %>% forecast(h = 8)

# Continent
results_continent = fc %>%
  filter(
    is_aggregated(Country)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_continent = results_continent %>% add_column(group = 'continent', .before=".model")
results = results_continent

# Country
results_continent = fc %>%
  filter(
    is_aggregated(Continent)
  ) %>%
  accuracy(data = police_gts, measures = list(
    mase = MASE,
    rmse = RMSE
  )) %>%
  group_by(.model) %>%
  summarise(mase = mean(mase), rmse = mean(rmse))
results_continent = results_continent %>% add_column(group = 'beat', .before=".model")
results = full_join(results, results_continent)

# Total
results_tot = fc %>%
  filter(
    is_aggregated(Continent), is_aggregated(Country)
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
    !is_aggregated(Continent), !is_aggregated(Country)
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

