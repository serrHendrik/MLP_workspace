start "SERVER" run_server.bat
start "AGENT CONTROLLER" run_agent_controller.bat
pause
cd the_apples_game
SET episodes=20
SET agent=ws://localhost:8081
@for /l %%x in (1, 1, %episodes%) do @(
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% >NUL)

cd ..

