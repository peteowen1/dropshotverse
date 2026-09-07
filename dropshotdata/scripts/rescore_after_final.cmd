@echo off
REM Settle arm C against arm D once the US Open late rounds exist.
REM
REM Registered as the scheduled task DropshotverseRescoreAfterFinal, one shot on
REM 2026-09-14, the day after the men's final. Same shape as the other two tasks
REM here: cmd /c ... >> logfile, so the outcome is readable in the morning.
REM
REM Writes nothing to blog/ and publishes nothing. It rebuilds the point table,
REM re-runs both in-match arms and prints the comparison. If the capture has not
REM collected the later rounds it says so and stops rather than restating what
REM is already on record.
cd /d C:\dev\dropshotverse\dropshotdata\scripts || exit /b 1
echo(
echo ==== rescore start %DATE% %TIME% ====
C:\Python313\python.exe rescore_after_final.py
if errorlevel 1 (
    echo RESCORE FAILED - see above
    exit /b 1
)
echo ==== rescore ok %DATE% %TIME% ====
