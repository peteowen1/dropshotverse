# Sanity-check + serve-stat coverage for Aneeshers/tennis-sackmann-archive
# Ticket 13. Plain CSV work -- no arrow/parquet involved, safe under Git Bash R.

suppressMessages(library(data.table))

root <- "C:/dev/dropshotverse/dropshotdata/raw/github"
out_dir <- "C:/dev/dropshotverse/dropshotdata/manifest"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

sink_file <- "C:/dev/dropshotverse/dropshotdata/manifest/sanity_check_output.txt"
sink(sink_file, split = TRUE)

cat("=== Sanity check run at", format(Sys.time(), tz = "UTC", usetz = TRUE), "===\n\n")

classify_file <- function(path) {
  fname <- basename(path)
  tour <- if (grepl("^atp_", fname)) "ATP" else if (grepl("^wta_", fname)) "WTA" else NA_character_
  file_group <- if (grepl("qual_chall", fname)) "qual_chall"
                else if (grepl("qual_itf", fname)) "qual_itf"
                else if (grepl("futures", fname)) "futures"
                else if (grepl("doubles", fname)) "doubles"
                else if (grepl("^(atp|wta)_matches_[0-9]{4}\\.csv$", fname)) "main"
                # atp_matches_amateur.csv: pre-Open-era ATP file (1877-1967), same 49-col schema,
                # no WTA equivalent (WTA wasn't founded until 1973). Real file, not noise -- treat
                # as "main"/tour level, same as the yearly Open-era files.
                else if (fname == "atp_matches_amateur.csv") "main"
                else NA_character_
  list(tour = tour, file_group = file_group)
}

# Restrict to the atp/ and wta/ directories only. Note: slam_pointbypoint/ also contains files
# named "*-matches*.csv" (e.g. "2015-frenchopen-matches.csv") -- a totally different schema
# (slam-specific point-by-point match metadata, not the 49-col Sackmann match schema). An earlier
# version of this script globbed those in by accident via a loose "_matches_" pattern and it
# corrupted the row counts/date ranges with garbage (implausible 1870s-1960s "ATP" rows). Fixed
# by scoping the glob to atp/ and wta/ explicitly.
match_files <- c(
  list.files(file.path(root, "atp"), pattern = "_matches_.*\\.csv$", full.names = TRUE),
  list.files(file.path(root, "wta"), pattern = "_matches_.*\\.csv$", full.names = TRUE)
)
match_files <- match_files[!grepl("doubles", match_files)]  # singles only for this project
cat("Found", length(match_files), "singles match-level CSV files (main + qual_chall/qual_itf + futures)\n\n")

read_one <- function(fp) {
  cls <- classify_file(fp)
  # fill=TRUE: atp_matches_amateur.csv (1877-1967, pre-Open-era) is genuinely ragged -- every
  # row has only 40 of the 49 declared columns (no ranking system existed pre-1973, and several
  # trailing loser-serve-stat columns are dropped too). fread right-pads with NA, which is exactly
  # correct here since those fields are legitimately absent, not corrupted.
  dt <- tryCatch(
    fread(fp, colClasses = "character", showProgress = FALSE, na.strings = c("", "NA"), fill = TRUE),
    error = function(e) { cat("  FAILED to read:", fp, "-", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(dt) || nrow(dt) == 0) return(NULL)
  dt[, source_file := basename(fp)]
  dt[, tour := cls$tour]
  dt[, file_group := cls$file_group]
  dt
}

all_list <- lapply(match_files, read_one)
all_list <- Filter(Negate(is.null), all_list)
cat("Successfully read", length(all_list), "files\n\n")

dt <- rbindlist(all_list, use.names = TRUE, fill = TRUE)
cat("=== RAW COMBINED ROW COUNT:", nrow(dt), "===\n\n")

# --- derive level (tour/challenger/futures/qualifying) ---
# Verified directly against the data (see ticket 13 answer for the check):
#   - main files ("*_matches_YYYY.csv"): every row is a main-tour-level match -> "tour"
#     (tourney_level in {G,M,A,F,D,O,...}, no qualifying rounds live here)
#   - futures files (ATP only): main-draw + qualifying of Futures/15k-25k events -> "futures"
#   - qual_chall / qual_itf files mix TWO different things, distinguished by `round`, not `tourney_level`:
#       * round starts with "Q" (Q1/Q2/.../Q5) -> a QUALIFYING-round match for an event of ANY level
#         (tourney_level C/A/G/M for ATP; C/10/15/.../100/W/P/PM/I/G/T1-5/E for WTA) -> "qualifying"
#       * round does NOT start with "Q" AND tourney_level marks a genuine lower-tier main draw
#         (ATP: "C"=Challenger; WTA: "C" or a numeric prize-money tier "10".."100") -> "challenger"
#   Confirmed zero rows where a non-Q round carries a main-tour-level code (A/G/M for ATP;
#   W/P/PM/I/G/T1-5/E for WTA) -- i.e. main draws of tour-level events never leak into this file.
is_qual_round <- function(r) grepl("^Q", toupper(r))
wta_lower_tier_codes <- c("C","10","15","20","25","35","40","50","60","75","80","100")

dt[, level := fifelse(
  file_group == "main", "tour",
  fifelse(file_group == "futures", "futures",
    fifelse(file_group == "qual_chall",
      fifelse(is_qual_round(round), "qualifying",
        fifelse(tourney_level == "C", "challenger", paste0("atp_qual_chall_unclassified:", tourney_level))),
      fifelse(file_group == "qual_itf",
        fifelse(is_qual_round(round), "qualifying",
          fifelse(tourney_level %in% wta_lower_tier_codes, "challenger", paste0("wta_qual_itf_unclassified:", tourney_level))),
        NA_character_))))]

cat("=== level classification check: any unclassified rows? ===\n")
print(dt[grepl("unclassified", level), .N, by = .(tour, file_group, tourney_level, round, level)])
cat("(WTA 'challenger' here means ITF-circuit main draw across all prize-money tiers 10k-100k plus the",
    "generic legacy 'C' code -- WTA has no separate Futures tier; this is its closest analog.)\n\n")

# --- tourney_date parsing + decade ---
dt[, tourney_date_parsed := as.IDate(tourney_date, format = "%Y%m%d")]
dt[, decade := paste0(substr(tourney_date, 1, 3), "0s")]

cat("=== Row counts by tour ===\n")
print(dt[, .N, by = tour][order(tour)])
cat("\n=== Row counts by tour x level ===\n")
print(dt[, .N, by = .(tour, level)][order(tour, level)])
cat("\n=== Row counts by tour x decade ===\n")
print(dcast(dt[, .N, by = .(tour, decade)], decade ~ tour, value.var = "N")[order(decade)])
cat("\n")

cat("=== Date range overall ===\n")
cat("min:", format(min(dt$tourney_date_parsed, na.rm = TRUE)), " max:", format(max(dt$tourney_date_parsed, na.rm = TRUE)), "\n\n")
cat("=== Date range by tour ===\n")
print(dt[, .(min_date = min(tourney_date_parsed, na.rm=TRUE), max_date = max(tourney_date_parsed, na.rm=TRUE), n = .N), by = tour])
cat("\n=== Date range by tour x level ===\n")
print(dt[, .(min_date = min(tourney_date_parsed, na.rm=TRUE), max_date = max(tourney_date_parsed, na.rm=TRUE), n = .N), by = .(tour, level)][order(tour, level)])
cat("\n")

cat("=== Unique player counts (winner_id union loser_id) ===\n")
cat("Overall:", uniqueN(c(dt$winner_id, dt$loser_id)), "\n")
print(dt[, .(unique_players = uniqueN(c(winner_id, loser_id))), by = tour])
cat("\n")

cat("=== Duplicate row check (same tourney_id + match_num + tour) ===\n")
dupes <- dt[, .N, by = .(tour, tourney_id, match_num)][N > 1]
cat("Groups with >1 row for same tourney_id+match_num+tour:", nrow(dupes), "\n")
cat("(match_num is not globally unique within a tourney_id -- qualifying rounds and the main\n",
    " draw of the same event can reuse the same match_num values under Sackmann's numbering,\n",
    " so this count is NOT proof of true duplicate rows. Re-checked including round + level:\n")
dupes2 <- dt[, .N, by = .(tour, tourney_id, match_num, round, level)][N > 1]
cat("Groups with >1 row for same tourney_id+match_num+round+level+tour (true collisions):", nrow(dupes2), "\n\n")

cat("=== Column population (fraction non-NA) across ALL columns, overall ===\n")
cols <- setdiff(names(dt), c("source_file","tour","file_group","level","decade","tourney_date_parsed"))
pop <- sapply(cols, function(cn) round(mean(!is.na(dt[[cn]]) & dt[[cn]] != ""), 4))
print(sort(pop))
cat("\n")

# ============================================================
# SERVE-STAT COVERAGE TABLE -- the ticket-12-critical deliverable
# ============================================================
serve_cols_w <- c("w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved")
serve_cols_l <- c("l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved")

to_num <- function(x) suppressWarnings(as.numeric(x))
for (cc in c(serve_cols_w, serve_cols_l)) {
  if (cc %in% names(dt)) dt[[cc]] <- to_num(dt[[cc]])
}

dt[, has_full_serve_line := Reduce(`&`, lapply(c(serve_cols_w, serve_cols_l), function(cc) {
  if (cc %in% names(dt)) !is.na(dt[[cc]]) else rep(FALSE, nrow(dt))
}))]

cat("=== SERVE-STAT COVERAGE: fraction of matches with a FULL serve line (svpt,1stIn,1stWon,2ndWon,bpSaved BOTH players non-null) ===\n\n")

cat("--- by tour ---\n")
print(dt[, .(n_matches = .N, n_with_serve = sum(has_full_serve_line), coverage = round(mean(has_full_serve_line), 4)), by = tour][order(tour)])

cat("\n--- by tour x level ---\n")
print(dt[, .(n_matches = .N, n_with_serve = sum(has_full_serve_line), coverage = round(mean(has_full_serve_line), 4)), by = .(tour, level)][order(tour, level)])

cat("\n--- by tour x decade ---\n")
print(dt[, .(n_matches = .N, n_with_serve = sum(has_full_serve_line), coverage = round(mean(has_full_serve_line), 4)), by = .(tour, decade)][order(tour, decade)])

cat("\n--- FULL BREAKDOWN: tour x level x decade ---\n")
full_tbl <- dt[, .(n_matches = .N, n_with_serve = sum(has_full_serve_line), coverage = round(mean(has_full_serve_line), 4)), by = .(tour, level, decade)][order(tour, level, decade)]
print(full_tbl, nrows = 200)

fwrite(full_tbl, file.path(out_dir, "serve_stat_coverage_by_tour_level_decade.csv"))

cat("\n--- per-column non-null rate within the matches that DO have a full serve line vs those that don't (sanity) ---\n")
for (cc in c(serve_cols_w, serve_cols_l)) {
  if (cc %in% names(dt)) cat(cc, ": ", round(mean(!is.na(dt[[cc]])), 4), "\n")
}

# also write summary tables as csv for the manifest
fwrite(dt[, .N, by = .(tour, level)][order(tour, level)], file.path(out_dir, "row_counts_by_tour_level.csv"))
fwrite(dt[, .N, by = .(tour, decade)][order(tour, decade)], file.path(out_dir, "row_counts_by_tour_decade.csv"))

cat("\n=== DONE ===\n")
sink()
cat("Sanity check complete. Output written to", sink_file, "\n")
