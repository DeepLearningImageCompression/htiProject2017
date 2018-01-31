@echo off
title PFE final exe

setlocal enableDelayedExpansion
for %%I in (./dataset/*.jpg) do (
  echo encodage %%I
  python resize_grey_img.py ./dataset %%I
  cd ./bpg
  set img=%%I
  echo !img!
  set sanscode=!img:~0,-3!
  echo !sanscode!
  echo ..\\results\\!sanscode!bpg
  bpgenc.exe -o ..\\results\\!sanscode!bpg -q 33 ../resize_grey.png
  echo decodage
  bpgdec.exe -o ../test_lr/!img! ..\\results\\!sanscode!bpg
  cd ..
  echo. )
endlocal

echo SRGAN
cd ./SRGAN_Brade
python main.py
cd ..

pause > nul
