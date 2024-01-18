set SYCL_ENABLE_DEFAULT_CONTEXTS=1
set SYCL_CACHE_PERSISTENT=1
call ".\oneAPI\compiler\2023.2.0\env\vars.bat"
call ".\oneAPI\mkl\2023.2.0\env\vars.bat"
audio\python.exe gpu_LLM_mtl_KOL.py