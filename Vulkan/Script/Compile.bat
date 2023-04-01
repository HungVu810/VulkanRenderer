@echo off
setlocal EnableDelayedExpansion

set compiler=..\Dependencies\glslc.exe
set shaderSourceDir=..\Shader
set shaderBinDir=..\Binaries\Shader

:: Program arguments
:: ------ Argument 1:
:: vert: Vertex shader (0)
:: tesc: Tessellation control shader (1)
:: tese: Tessellation evaluation shader (2)
:: geom: Geometry shadr (3)
:: frag: Fragment shader (4)
:: comp: Compute shader (5)
:: ------ Argument 2: Name of the shader in %shaderSourceDir%, taking form as shaderName.glsl

del %shaderBinDir%\*
%compiler% -fshader-stage=vert %shaderSourceDir%\Vertex.glsl -o %shaderBinDir%\Vertex.spv
%compiler% -fshader-stage=frag %shaderSourceDir%\Fragment.glsl -o %shaderBinDir%\Fragment.spv

pause

