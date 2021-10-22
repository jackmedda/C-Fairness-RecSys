library(readr)
library(dplyr)

message("reading users")
users = read_tsv("data/lastfm-dataset-360K/usersha1-profile.tsv",
                 col_names=c("key", "gender", "age", "country"),
                 col_types="ccic_", progress=TRUE) %>%
    mutate(id=1:n())
message(sprintf("read %d users", nrow(users)))

message("reading records")
records = read_tsv("data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv",
                   col_names=c("user", "artid", "plays"),
                   col_types="cc_i", progress=TRUE)
message(sprintf("read %d records", nrow(records)))

message("splitting out artists")
artist_ids = unique(records$artid)
artists = data.frame(id=1:length(artist_ids), artid=artist_ids)
message(sprintf("extracted %d artists", nrow(records)))

message("joining plays")
play_counts = records %>%
    inner_join(users %>% select(uid=id, user=key)) %>%
    inner_join(artists) %>%
    select(user=uid, item=id, count=plays) %>%
    filter(!is.na(count))
message(sprintf("have %d plays of %d artists by %d users",
                nrow(play_counts), nrow(artists), nrow(users)))

message("writing data")
write_csv(users %>% select(id, key, gender, age),
          "build/lastfm-users.csv", na="")
write_csv(artists, "build/lastfm-artists.csv", na="")
write_csv(play_counts, "build/lastfm-play-counts.csv", na="")

message("partitioning data")
user.counts = play_counts %>%
    group_by(user) %>%
    summarize(nratings = n())
item.counts = play_counts %>%
    group_by(item) %>%
    summarize(nratings = n())

test_users = user.counts %>%
    ungroup() %>%
    filter(nratings >= 10) %>%
    sample_n(25000) %>%
    mutate(partition=rep(1:5, 5000))

test_user_ratings = play_counts %>%
    inner_join(test_users) %>%
    group_by(user) %>%
    mutate(rank = sample(n())) %>%
    filter(rank <= 5)

test_items = user.counts %>%
    ungroup() %>%
    filter(nratings >= 10) %>%
    sample_n(25000) %>%
    mutate(partition=rep(1:5, 5000))

test_item_ratings = play_counts %>%
    inner_join(test_items) %>%
    group_by(item) %>%
    mutate(rank = sample(n())) %>%
    filter(rank <= 5)

dir.create("build/lastfm-splits", showWarnings=FALSE)
for (i in 1:5) {
    message(sprintf("writing partition %d", i))
    testr = test_user_ratings %>% filter(partition == i)
    trainr = play_counts %>% left_join(select(testr, user, item, partition)) %>% filter(is.na(partition)) %>% select(user, item, count)
	print(trainr)
    write_csv(testr %>% select(user, item, count),
              sprintf("build/lastfm-splits/u.part%d.test.csv", i), na="")
    write_csv(trainr, sprintf("build/lastfm-splits/u.part%d.train.csv", i), na="")
    testr = test_item_ratings %>% filter(partition == i)
    trainr = play_counts %>% left_join(select(testr, user, item, partition)) %>% filter(is.na(partition)) %>% select(user, item, count)
	print(trainr)
    write_csv(testr %>% select(user, item, count),
              sprintf("build/lastfm-splits/i.part%d.test.csv", i), na="")
    write_csv(trainr, sprintf("build/lastfm-splits/i.part%d.train.csv", i), na="")
	break
}
