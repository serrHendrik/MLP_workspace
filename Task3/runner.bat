start "AGENT CONTROLLER" run_agent_IA_controller_PLAY.bat
SET agent=ws://127.0.0.1:8081
pause
cd the_apples_game
SET episodes=50
SET apples=10
@for /l %%x in (1, 1, %episodes%) do @(
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %apples%>NUL)

cd ..

