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
#from collections import defaultdict
#import random
from tensorforce.agents import DQNAgent
import numpy as np


logger = logging.getLogger(__name__)
games = {}
agentclass = None

"""
The Agent_IA is Inequity Averse (IA), both advantageous (AIA) and disadvantageous (DIA)
"""
class Agent_IA:
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
        
        #Keep one DQNAgent per player controlled by Agent_IA
        self.DQNAgents = dict([(player,self.get_new_DQNAgent())])
        
        #define the mapping of the action scalar to a certain action
        self.actions = ["left","move","right","fire"]
        
        #Data structures for Inequity Aversion
        #rewards = scores_t - scores_tMinus1
        self.scores_tMinus1 = np.zeros(nb_players)
        self.scores_t = np.zeros(nb_players)
        self.rewards = np.zeros(nb_players)
        #e values and subjective rewards as defined in (Hughes, Leibo, Tuyls et al. 2018)
        self.g = 0.9    #gamma used to calculate e-value
        self.l = 1      #lamda used to calculate e-value
        self.e_values = np.zeros(nb_players)
        self.a = 5.0    #alpha used to calculate subjective rewards
        self.b = 0.05   #beta used to calculate subjective rewards
        self.subj_rewards = np.zeros(nb_players)
        
        

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)
        self.DQNAgents[player] = self.get_new_DQNAgent()

    def register_action(self, row, column, orientation, player):
        """Register action played in game.
        :param row:
        :param columns:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        pass
    
    def observe(self, j, score, terminal):
        # Change score for player j
        self.scores_tMinus1[j-1] = self.scores_t[j-1]
        self.scores_t[j-1] = score
        
        #If j is a player of this agent, update model
        if j in self.player:
            subj_reward = self.calc_subj_reward(j)
            self.DQNAgents[j].observe(reward=subj_reward, terminal=terminal)
    
    def next_action(self, player, players, apples):
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))
        # Random move
        #return 'move'
        
        state = np.zeros((15,15,3))
        player_x, player_y = players[player-1]["location"]
        print("Current Player: " + str(player))
        print(str(players[player-1]))
        #print("player_x and player_y: " + str(player_x) + " " + str(player_y))
        #fill network input with apples on second channel
        for apple_x, apple_y in apples:
            #print("apple_x and apple_y original: " + str(apple_x) + " " + str(apple_y))
            if apple_x < player_x - 7:
                apple_x += self.nb_cols
            elif apple_x > player_x + 7:
                apple_x -= self.nb_cols
            
            if apple_y < player_y - 7:
                apple_y += self.nb_rows
            elif apple_y > player_y + 7:
                apple_y -= self.nb_rows
            #print("apple_x and apple_y transformed: " + str(apple_x) + " " + str(apple_y))
            inp_x = apple_x - player_x + 7 - 1
            inp_y = apple_y - player_y + 7 - 1
            #print("inp_x and inp_y: " + str(inp_x) + " " + str(inp_y))
            state[inp_y,inp_x,1] = 1
        
        #fill network ipnut with players on third channel
        for p in players:
            p_x, p_y = p["location"]
            if (type(p_x) == int):
                #handle endless field
                #print("p_x and p_y original: " + str(p_x) + " " + str(p_y))
                if p_x < player_x - 7:
                    p_x += self.nb_cols
                elif p_x > player_x + 7:
                    p_x -= self.nb_cols
                
                if p_y < player_y - 7:
                    p_y += self.nb_rows
                elif p_y > player_y + 7:
                    p_y -= self.nb_rows
                #print("p_x and p_y transformed: " + str(p_x) + " " + str(p_y))
                inp_x = p_x - player_x + 7 - 1
                inp_y = p_y - player_y + 7 - 1
                state[inp_y,inp_x,2] = 1
        
        # Exploit symmetry to help network train better
        # Rotate state so that player orientation is up
        orientation = players[player-1]["orientation"]
        
        if orientation == "right":
            state = np.rot90(state,k=1)
        elif orientation == "down":
            state = np.rot90(state,k=2)
        elif orientation == "left":
            state = np.rot90(state,k=3)
        
        #Feed to network
        action_index = self.DQNAgents[player].act(states=state)
        return self.actions[action_index]
        

    def end_game(self):
        self.ended = True
        # store models
        for p in self.player:
            self.DQNAgents[p].save_model(directory="../models/")
    
    # TODO: Add preprocessing (gray scale) + greedy-epsilon exploration
    def get_new_DQNAgent(self):
        #batch_size = 4096
        states_spec = dict(shape=(15,15,3), type='float')
        actions_spec = dict(num_actions=4, type='int')
        network_spec = [
            dict(type='flatten'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh')
        ]
        agent = DQNAgent(
                states=states_spec,
                actions=actions_spec,
                network=network_spec,
                #memory=dict(
                #    type='replay', 
                #    include_next_states=False, 
                #    capacity=1000*batch_size
                #),
        )
        
        agent.reset()
        
        return agent

    def calc_subj_reward(self, i):   
        # Update rewards
        self.rewards = self.scores_t - self.scores_tMinus1
        
        # Update e values
        self.e_values = self.g * self.l * self.e_values + self.rewards
        
        # Calculdate subjective reward
        player_index = i - 1
        envy = 0.0
        guilt = 0.0
        for j in range(0,self.nb_players):
            if self.e_values[j] > self.e_values[player_index]:
                envy += self.e_values[j] - self.e_values[player_index]
            else:
                guilt += self.e_values[player_index] - self.e_values[j]
        
        envy = envy * self.a / float(self.nb_players - 1)
        guilt = guilt * self.b / float(self.nb_players - 1)
        
        sr = self.rewards[player_index] - envy - guilt
        return sr
            


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
                    #nm = games[game].next_action(msg["player"], msg["players"], msg["apples"])
                    nm = 'move'
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
                # Let agent have knowledge of most recent action and new score of player who played
                j = msg["player"]
                score = msg["players"][j-1]["score"]
                games[game].observe(j, score, False)
            
                if msg["nextplayer"] in games[game].player:
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

    agentclass = Agent_IA
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
