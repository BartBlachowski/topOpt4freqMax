@echo off
cd /d "C:\Users\ptauzow\topopt_revision_0166fef\examples\Revision_applsci"
echo START %DATE% %TIME% > run_alpha_sweep_building.log
start "" /low /wait /b "D:\MATLAB\R2024a\bin\matlab.exe" -batch "launch_building" -logfile run_alpha_sweep_building_matlab.log
echo EXIT=%ERRORLEVEL% >> run_alpha_sweep_building.log
echo END %DATE% %TIME% >> run_alpha_sweep_building.log
