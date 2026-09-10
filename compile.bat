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
echo Run it directly, or add target\release to your PATH:
echo   target\release\fastdna.exe --input reads.fastq --output counts.parquet
echo.
echo For the Python extension instead of the CLI, use maturin:
echo   pip install maturin
echo   maturin develop --release --features python
echo.
pause