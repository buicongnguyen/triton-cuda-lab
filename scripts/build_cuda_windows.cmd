@echo off
setlocal
rem Builds build\cuda_portfolio.exe from any directory. Requires VS C++ tools and
rem CUDA_PATH from the Toolkit. Set CUDA_ARCH (e.g. sm_86) for a GPU other than sm_89.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere was not found: install Visual Studio with the "Desktop development with C++" workload. & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (echo Visual Studio C++ x64 tools were not found: add the "Desktop development with C++" workload. & exit /b 1)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not defined CUDA_PATH (echo CUDA_PATH is not set: install the CUDA Toolkit, then open a new terminal. & exit /b 1)
if not defined CUDA_ARCH set "CUDA_ARCH=sm_89"
pushd "%~dp0.."
if not exist build mkdir build
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 -lineinfo -arch=%CUDA_ARCH% cuda\kernels.cu -o build\cuda_portfolio.exe
set "STATUS=%errorlevel%"
popd
exit /b %STATUS%
