@echo off
title FastDNA Engine - Build Release

echo ==================================================
echo   FastDNA: Building Engine in Release Mode
echo ==================================================
echo.

:: Run the optimized build
cargo build --release

:: Check if the build failed
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed. Please check the Rust errors above.
    echo ==================================================
    pause
    exit /b %errorlevel%
)

echo.
echo ==================================================
echo   [SUCCESS] Engine compiled successfully.
echo ==================================================
echo.
echo Your optimized executable is ready at:
echo   -\target\release\fastdna.exe
echo.
echo [Next Step]: 
echo Copy that "fastdna.exe" file and paste it into the "bin/" folder
echo of your other project (fastdna-research) to use it with Python and Docker.
echo.
pause