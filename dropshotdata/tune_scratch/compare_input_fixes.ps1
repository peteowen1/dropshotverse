param([string]$scale = "0.0")
$fix = "C:\dev\dropshotverse\dropshotdata\tune_scratch\elo_raw_ratings_full.SURFACEFIX.parquet"
$legacy = "C:\dev\dropshotverse\dropshotdata\tune_scratch\elo_raw_ratings_full.parquet"
$arms = @(@("old",$legacy,"1.0"), @("elo",$fix,"1.0"), @("hier",$legacy,$scale), @("both",$fix,$scale))
foreach ($a in $arms) {
  $env:DSV_ELO_RAW_PATH = $a[1]; $env:DSV_SURF_RW_SCALE = $a[2]
  C:\Python313\python.exe C:\dev\dropshotverse\dropshotdata\tune_scratch\compare_input_fixes.py $a[0] $a[2] 2>&1
}
