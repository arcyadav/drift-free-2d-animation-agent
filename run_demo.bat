@echo off
REM Live demo for the viva: starts the agent with a public share link.
cd /d %~dp0
C:\Users\ysarc\.venvs\aids_p3\Scripts\python.exe app.py --share
pause
