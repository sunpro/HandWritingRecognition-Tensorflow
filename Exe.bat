@echo off
if exist ".\HandWritingFileName.txt" (del  ".\HandWritingFileName.txt")
start python Input.py
loopstart: 
if exist ".\HandWritingFileName.txt" goto readname
goto loopstart
readname:
start python ImageProcess.py

pause