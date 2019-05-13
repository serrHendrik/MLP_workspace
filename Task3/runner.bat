start "AGENT CONTROLLER" run_agent_FR_controller_TRAIN.bat
SET agent=ws://127.0.0.1:8082
pause
cd the_apples_game
SET episodes=50
SET apples=50
@for /l %%x in (1, 1, %episodes%) do @(
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %apples% >NUL
)

cd ..

