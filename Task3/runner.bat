start "AGENT FR CONTROLLER" run_agent_FR_controller_TRAIN.bat
pause
cd the_apples_game
SET episodes=2
SET agent=ws://127.0.0.1:8082
SET apples=10
@for /l %%x in (1, 1, %episodes%) do @(
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)

cd ..

