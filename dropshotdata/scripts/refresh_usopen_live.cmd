@echo off
REM Refresh the live US Open projection and publish it to R2.
REM
REM Driven by the Windows scheduled task DropshotverseUSOpenLiveSim, which
REM mirrors DropshotverseUSOpenCapture's shape (cmd.exe /c ... >> logfile).
REM
REM Ordering matters: if the simulation fails we must NOT run the upload, or a
REM stale parquet gets re-published with a fresh updated_at and silently looks
REM current. The errorlevel guard below is the whole point of this wrapper.

cd /d C:\dev\dropshotverse\dropshotdata\scripts || exit /b 1

echo(
echo ==== refresh start %DATE% %TIME% ====

C:\Python313\python.exe simulate_usopen_live.py --sims 20000
if errorlevel 1 (
    echo SIMULATION FAILED - skipping upload, keeping previously published data
    exit /b 1
)

C:\Python313\python.exe upload_blog_data.py
if errorlevel 1 (
    echo UPLOAD FAILED
    exit /b 1
)

echo ==== refresh ok %DATE% %TIME% ====
