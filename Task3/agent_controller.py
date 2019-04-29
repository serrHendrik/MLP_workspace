#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.
Copyright (c) 2018 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
import numpy as np
from agent_IA import agent_IA
from agent_free_rider import agent_free_rider


logger = logging.getLogger(__name__)
games = {}
agentclass = None

"""
The Agent_IA is Inequity Averse (IA), both advantageous (AIA) and disadvantageous (DIA)
"""
class Agent_controller:
    """
    A Agent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game
    This class does not necessarily use the best data structures for the
    approach you want to use.
    """
    def __init__(self, player, nb_rows, nb_cols, nb_players):
        """Create Dots and Boxes agent.
        :param player: Player number, 1 or 2
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        """
        self.player = {player}
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        
        #define the mapping of the action scalar to a certain action
        self.actions = ["left","move","right","fire"]
        
        # Total number of inequity adverse agents
        self.IA_agents_allowed = 2
        self.IA_agents_present = 0
        
        #Keep one agent per player
        self.agents = dict()
        if self.IA_agents_present < self.IA_agents_allowed:
            self.agents[player] = agent_IA(player, nb_rows, nb_cols, nb_players, self.actions)
            self.IA_agents_present += 1
        else:
            self.agents[player] = agent_free_rider(player, nb_rows, nb_cols, nb_players, self.actions)
        
        #Data structures for rewards calculations
        #rewards = scores_t - scores_tMinus1
        self.scores_tMinus1 = np.zeros(nb_players)
        self.scores_t = np.zeros(nb_players)
        self.rewards = np.zeros(nb_players)
        
        

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)
        if self.IA_agents_present < self.IA_agents_allowed:
            self.agents[player] = agent_IA(player, self.nb_rows, self.nb_cols, self.nb_players, self.actions)
            self.IA_agents_present += 1
        else:
            self.agents[player] = agent_free_rider(player, self.nb_rows, self.nb_cols, self.nb_players, self.actions)

    def register_action(self, row, column, orientation, player):
        """Register action played in game.
        :param row:
        :param columns:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        pass
    
    def observe(self, scores, terminal):
        # Update scores
        self.scores_tMinus1 = self.scores_t
        self.scores_t = scores
        self.rewards = self.scores_t - self.scores_tMinus1
        
        #For all players controlled by this agent controller, push updates
        for p in self.player:
            self.agents[p].observe(self.rewards, terminal)

    
    def next_action(self, player, players, apples):
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))
        
        return self.agents[player].next_action(players,apples)
        

    def end_game(self):
        self.ended = True
        # store models
        for p in self.player:
            self.agents[p].save_model(directory="../models/")
    


            


## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    game = None
    # msg = await websocket.recv()
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None
            #print("JSON MSG:\n" + str(msg) + "\n")
            if msg["type"] == "start":
                # Initialize game
                if msg["game"] in games:
                    games[msg["game"]].add_player(msg["player"])
                else:
                    nb_cols, nb_rows = msg["grid"]
                    games[msg["game"]] = agentclass(msg["player"],
                                                    nb_rows,
                                                    nb_cols,
                                                    len(msg["players"]))
                
                if msg["player"] == 1:
                    # Start the game
                    nm = games[game].next_action(msg["player"], msg["players"], msg["apples"])
                    print('nm = {}'.format(nm))
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    answer = {
                        'type': 'action',
                        'action': nm,
                    }
                else:
                    # Wait for the opponent
                    answer = None

            elif msg["type"] == "action":
                # An action has been played
                # Let agent have knowledge of new scores at the beginning of new time step
                if msg["nextplayer"] == 1 and msg["receiver"] == min(games[game].player):
                    scores = np.zeros(len(msg["players"]))
                    for j in range(0,len(msg["players"])):
                        scores[j] = msg["players"][j]["score"]
                    games[game].observe(scores, False)
            
                if msg["nextplayer"] in games[game].player and msg["nextplayer"] == msg["receiver"]:
                    # Compute your move
                    nm = games[game].next_action(msg["nextplayer"], msg["players"], msg["apples"])
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    answer = {
                        'type': 'action',
                        'action': nm
                    }
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                print(answer)
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass
    parser = argparse.ArgumentParser(description='Start agent to play the Apples game')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = Agent_controller
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
