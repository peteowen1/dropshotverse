$env:DSV_EVAL_YEARS = "2022,2023"
$env:DSV_RW_VAR_PER_WEEK="0.005"; $env:DSV_SURFACE_VAR_SCALE="1.0"; $env:DSV_OU_THETA_PER_WEEK="0.00001"; $env:DSV_MAX_SHIFT="0.125"
$env:DSV_PHI_MAX="4.0"; $env:DSV_LAYOFF_C="0.06"; $env:DSV_REGIME_B_N="0.25"; $env:DSV_TIER_WEIGHT="1.0"; $env:DSV_CLIP_REINFLATE_K="0.5"
foreach ($s in "1.0","0.3","0.1","0.0") {
  $env:DSV_SURF_RW_SCALE = $s
  $env:DSV_OUT_PREDS = "C:\dev\dropshotverse\dropshotdata\tune_scratch\surf_rw_v25_$s`_preds.parquet"
  $env:DSV_OUT_STATE = "C:\dev\dropshotverse\dropshotdata\tune_scratch\surf_rw_v25_$s`_state.pkl"
  Write-Output "=== v25 params, SURF_RW_SCALE=$s ==="
  C:\Python313\python.exe C:\dev\dropshotverse\dropshotdata\scripts\hierarchical_model.py 2>&1 | Select-Object -Last 3
}
