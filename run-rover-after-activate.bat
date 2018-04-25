pushd "%~dp0\..\Windows_Roversim"
start .\Roversim_x86_64.exe
popd

pushd "%~dp0\code"
python --version
python drive_rover.py ../output/ 2>&1 > ..\logs\drive_rover.log & type ..\logs\drive_rover.log
popd

pushd "%~dp0"
python tools/convert_log.py logs/drive_rover.log ../note/rocks.json
python tools/make_videos.py

rem copy /Y output\worldmap.mp4 ..\note\images\worldmap.mp4

popd