cd the_apples_game
FOR /L %%A IN (1,1,100) DO (
  node play.js ws://localhost:9090 ws://localhost:9090 ws://localhost:9090 ws://localhost:9090 ws://localhost:9090 ws://localhost:9090 3
)