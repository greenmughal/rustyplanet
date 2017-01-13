@echo off
set FXC="%DXSDK_DIR%\Utilities\bin\x64\fxc.exe" -nologo
if not exist data mkdir data
%FXC% /T vs_5_0 /E BezierVS /Fo data/vertex.fx shader/terrain2.hlsl
%FXC% /T hs_5_0 /E BezierHS /Fo data/hull.fx shader/terrain2.hlsl
%FXC% /T ds_5_0 /E BezierDS /Fo data/domain.fx shader/terrain2.hlsl
%FXC% /T ps_5_0 /E BezierPS /Fo data/pixel.fx shader/terrain2.hlsl
