start "AGENT CONTROLLER" random_agent_run.bat
pause
cd the_apples_game
SET episodes=5
SET agent=ws://127.0.0.1:8080
@for /l %%y in (5, 5, 20) do @(
SET apples=%%y
@echo apples: %apples%
@for /l %%x in (1, 1, %episodes%) do @(
@echo 5 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
@for /l %%x in (1, 1, %episodes%) do @(
@echo 6 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
@for /l %%x in (1, 1, %episodes%) do @(
@echo 7 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
@for /l %%x in (1, 1, %episodes%) do @(
@echo 8 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
@for /l %%x in (1, 1, %episodes%) do @(
@echo 9 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
@for /l %%x in (1, 1, %episodes%) do @(
@echo 10 agents
@echo episode %%x / %episodes%
@node play.js %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %agent% %apples%>NUL)
)
cd ..