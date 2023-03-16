@echo off
setlocal EnableDelayedExpansion

set compiler=C:\VulkanSDK\1.3.236.0\Bin\glslc.exe
set shaderSourceDir=..\Shader
set shaderBinDir=..\Binaries\Shader

:: vert: Vertex shader
:: tesc: Tessellation control shader
:: tese: Tessellation evaluation shader
:: geom: Geometry shadr
:: frag: Fragment shader
:: comp: Compute shader

del %shaderBinDir%\*
%compiler% -fshader-stage=vert %shaderSourceDir%\Vertex.glsl -o %shaderBinDir%\Vertex.spv
%compiler% -fshader-stage=frag %shaderSourceDir%\Fragment.glsl -o %shaderBinDir%\Fragment.spv

pause

