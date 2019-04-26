Apples game application
==========================
Several agents roam a shared world and collect apples to receive positive rewards. They may also direct a beam at one of the other agents, “tagging them”, to temporarily remove them from the game, but this action does not trigger a reward.
<https://deepmind.com/blog/understanding-agent-cooperation/>

Live demo: <https://people.cs.kuleuven.be/~wannes.meert/the_apples_game/dist/>

![Screenshot of the Apples Game](https://people.cs.kuleuven.be/wannes.meert/the_apples_game/screenshot.png?v=2)

This setup is part of the course "Machine Learning: Project" (KU Leuven,
Faculty of engineering, Department of Computer Science,
[DTAI research group](https://dtai.cs.kuleuven.be)).


Installation
------------

The example agent is designed for Python 3.6 and requires the
[websockets](https://websockets.readthedocs.io) package. Dependencies can be
installed using pip:

    $ pip install -r requirements.txt


Start the game GUI
------------------

This program shows a web-based GUI to play the Apples
game. This supports human-human, agent-human and agent-agent combinations.
It is a simple Javascript based application that runs entirely in the browser.
You can start it by opening the file `dist/index.html` in a browser.
Or alternatively, you can start the app using the included simple server:

    $ ./server.py 8080

The game can then be played by directing your browser to http://127.0.0.1:8080.

Alternatively, you could run the game headless from the CLI. Therefore,
you should run the following command:

    $ node play.js ws://localhost:8001 ws://localhost:8002

The arguments are the websocket address of a game-playing agent. These agents
are described in the next section.

Start the agent client
----------------------

This is the program that runs a game-playing agent. This application listens
to [websocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
requests that communicate game information and sends back the next action it
wants to play.

Starting the agent client is done using the following command:

    $ ./agent.py <port>

This starts a websocket on the given port that can receive JSON messages.

The JSON messages given below should be handled by your agent.

### Initiate the game

Each agent gets a message that a new game has started:

    {
        "type": "start",
        "player": 1,
        "game": "123456",
        "grid": [36, 16],
        "players": [
          {
            "location": [7, 10],
            "orientation": "right"
          },
          {
            "location": ["?", "?"],
            "orientation": "?"
          }
        ],
        "apples": [[10, 15], [3, 5], ...]
    }

where `player` is the number assigned to this agent
and `grid` is the grid size in columns and rows. Furthermore, `players`
contains the initial locations and orientations of all players that participate. 
Finally, `apples` contains the locations of the apples. Locations
are represented as `[x, y]` coordinates with the origin at the top left. Note
that only the locations of apples that are within a 15x15 window of the agent's
location are included in the message. Similarly, the location and orientation of
the players that are outside this window are replaced with a question mark.

If you are player 1, reply with the first action you want to perform:

    {
        "type": "action",
        "action": "move"
    }

The field `orientation` is either 'move' (move on position forward), 'left' (turn left)
'right' (turn right) or 'fire' (fire at one the other agents).


### Action in the game

When an action is played, each agent receives a message of the following format:

    {
        "type": "action",
        "player": 1,
        "nextplayer": 2,
        "game": "123456",
        "players": [
          {
            "location": [7, 10],
            "orientation": "right",
            "score": 10
          },
          {
            "location": [33, 10],
            "orientation": "left",
            "score": 5
          }
        ],
        "apples": [[25, 15], [25, 14], ...]
    }

However, the fields `players` and `apples` will be different for each receiving agent,
depending on each agent's 15x15 observable window.

If it is your turn you should answer with a message that states your next
move:

    {
        "type": "action",
        "action": "fire"
    }


### Game end

When the game ends after an action, the message is slightly altered:

    {
        "type": "end",
        "game": "123456",
        "player": 1,
        "nextplayer": 0,
        "players": [
          {
            "location": [7, 10],
            "orientation": "right",
            "score": 10
          },
          {
            "location": [33, 10],
            "orientation": "left",
            "score": 5
          }
        ],
        "apples": [[25, 15], [25, 14], ...]
        "winner": 1
    }

The `type` field becomes `end` and a new field `winner` is set to the player
that has won the game.

Modify the agent client
-----------------------

The agent client is written using the [Phaser 3](https://phaser.io/phaser3) framework. A pre-compiled version is available in the `dist` folder. However, if you would like to modify this client you have to install the required dependencies:

```
npm install --global gulp-cli
npm install
```

Next use the commands `gulp` to start the development server and `gulp dist` to create a new compiled version in the `dist` folder. 


Contact information
-------------------

Main developer:

- Pieter Robberechts, <https://people.cs.kuleuven.be/pieter.robberechts>

Team:

- Wannes Meert, <https://people.cs.kuleuven.be/wannes.meert>
- Karl Tuyls, <http://www.karltuyls.net>
- Sebastijan Dumančić, <https://people.cs.kuleuven.be/sebastijan.dumancic>
- Robin Manhaeve, <https://people.cs.kuleuven.be/robin.manhaeve>

