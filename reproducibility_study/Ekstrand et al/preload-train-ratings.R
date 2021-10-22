library(purrr)
library(dplyr)
library(readr)

message("Loading ML1M data")
ml1m.train.ratings = map_df(1:5, function(part) {
    message("reading part ", part)
    test.fn = sprintf("build/ML-1M.out/part0%d.test.csv", part)
    train.fn = sprintf("build/ML-1M.out/part0%d.train.csv", part)
    test = suppressMessages(read_csv(test.fn, col_names=c("user", "item", "rating", "timestamp")))
    train = suppressMessages(read_csv(train.fn, col_names=c("user", "item", "rating", "timestamp")))
    test %>%
        select(user) %>%
        distinct() %>%
        mutate(part=part) %>%
        inner_join(train)
})
ml1m.train.ratings
save(ml1m.train.ratings, file="build/ml1m-train.Rdata")

message("Loading LFM360K")
lfm360k.train.counts = map_df(1:5, function(part) {
    message("reading part ", part)
    test.fn = sprintf("build/lastfm-splits/u.part%d.test.csv", part)
    train.fn = sprintf("build/lastfm-splits/u.part%d.train.csv", part)
    test = suppressMessages(read_csv(test.fn))
    train = suppressMessages(read_csv(train.fn))
    test %>%
        select(user) %>%
        distinct() %>%
        mutate(part=part) %>%
        inner_join(train)
})
message("finished loading LFM360K")
lfm360k.train.counts
save(lfm360k.train.counts, file="build/lfm360k-train.Rdata")

message("Loading LFM1K")
lfm1k.train.counts = map_df(1:5, function(part) {
    message("reading part ", part)
    test.fn = sprintf("build/lfm1k-splits/u.part%d.test.csv", part)
    train.fn = sprintf("build/lfm1k-splits/u.part%d.train.csv", part)
    test = suppressMessages(read_csv(test.fn))
    train = suppressMessages(read_csv(train.fn))
    test %>%
        select(user) %>%
        distinct() %>%
        mutate(part=part) %>%
        inner_join(train)
})
message("finished loading LFM1K")
lfm1k.train.counts
save(lfm1k.train.counts, file="build/lfm1k-train.Rdata")
