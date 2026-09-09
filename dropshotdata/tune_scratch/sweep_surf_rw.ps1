$env:DSV_EVAL_YEARS = "2022,2023"
foreach ($s in "1.0","0.3","0.1","0.0") {
  $env:DSV_SURF_RW_SCALE = $s
  $env:DSV_OUT_PREDS = "C:\dev\dropshotverse\dropshotdata\tune_scratch\surf_rw_$s`_preds.parquet"
  $env:DSV_OUT_STATE = "C:\dev\dropshotverse\dropshotdata\tune_scratch\surf_rw_$s`_state.pkl"
  Write-Output "=== SURF_RW_SCALE=$s ==="
  C:\Python313\python.exe C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py 2>&1 | Select-Object -Last 4
}
