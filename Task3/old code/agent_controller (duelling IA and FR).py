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
import csv
import datetime
import tensorflow as tf

from agent_IA import agent_IA
from agent_FR import agent_FR
from Keras_DQNAgent import Keras_DQNAgent
from Keras_DDQNAgent import Keras_DDQNAgent

logger = logging.getLogger(__name__)
games = {}
agentclass = None

#Statistics
now = datetime.datetime.now()
stats_file = "statistics/stats_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
header_written = False
def write_header(agents_IA, agents_FR):
    if header_written == False:
        with open(stats_file, 'w',newline='') as wf:
            writer = csv.writer(wf)
            l1 = ["IA" for _ in agents_IA]
            l2 = ["FR" for _ in agents_FR]
            l3 = ["","IA MEAN", "IA VARIANCE", "FR MEAN", "FR VARIANCE"]
            l = l1 + l2 + l3
            writer.writerow(l)
        wf.close()

def set_headers_written():
    global header_written
    header_written = True

def write_scores(scores_agents_IA, scores_agents_FR):
    #never modify header again
    set_headers_written()
    
    #row = scores per agent type and averages per type
    scores = scores_agents_IA + scores_agents_FR
    IA_mean = -1
    IA_var = -1
    FR_mean = -1
    FR_var = -1
    if len(scores_agents_IA) > 0:
        IA_mean = float(sum(scores_agents_IA)) / len(scores_agents_IA)
        IA_var = np.var(scores_agents_IA)
    if len(scores_agents_FR) > 0:
        FR_mean = float(sum(scores_agents_FR)) / len(scores_agents_FR)
        FR_var = np.var(scores_agents_FR)
    row = scores + ["", IA_mean, IA_var, FR_mean, FR_var]
    with open(stats_file, 'a',newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(row)
    wf.close()  


"""
Controller of all agents.

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
        
        #Handle gpu issues
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        tf.Session(config=config)
        
        self.player = {player}
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        self.state_size = 15*15
        
        #define the mapping of the action scalar to a certain action
        #self.actions = ["left","move","right","fire"]
        self.actions = ["left","move","right"]
        self.action_size = len(self.actions)
        
        # Total number of inequity adverse agents
        self.IA_agents_allowed = 5
        self.IA_agents_present = 0
        
        """
        #DQN
        #DQN network for Inequity Adverse (IA) agents!
        IA_model_name = "models/DQN_IA_MODEL"
        self.IA_network = Keras_DQNAgent(state_size = self.state_size, action_size = self.action_size, model_name=IA_model_name)
        #DQN network for Free Rider agents!
        FR_model_name = "models/DQN_FR_MODEL"
        self.FR_network = Keras_DQNAgent(state_size = self.state_size, action_size = self.action_size, model_name=FR_model_name)
        """
        
        #DDQN
        IA_model_name = "models/DDQN_MODEL"
        self.IA_network = Keras_DDQNAgent(state_size = self.state_size, action_size = self.action_size, model_name=IA_model_name)
        #DQN network for Free Rider agents!
        FR_model_name = "models/DDQN_MODEL"
        self.FR_network = Keras_DDQNAgent(state_size = self.state_size, action_size = self.action_size, model_name=FR_model_name)
        
        #Keep one agent per player and remember which kind of agent this is
        self.agents = dict()
        self.agents_IA = list()
        self.agents_FR = list()
        if self.IA_agents_present < self.IA_agents_allowed:
            self.agents[player] = agent_IA(self.IA_network, player, nb_rows, nb_cols, nb_players, self.state_size, self.action_size)
            self.IA_agents_present += 1
            self.agents_IA.append(player)
        else:
            self.agents[player] = agent_FR(self.FR_network, player, nb_rows, nb_cols, nb_players, self.state_size, self.action_size)
            self.agents_FR.append(player)
        
        #Data structures for rewards calculations
        #rewards = scores_t - scores_tMinus1
        self.scores_tMinus1 = np.zeros(nb_players)
        self.scores_t = np.zeros(nb_players)
        self.rewards = np.zeros(nb_players)
        # To shift scores once per timestep (= every player moved), use a dirty_counter
        # This dirty_counter makes sure that every player received an update with the current scores
        # before updating the scores once more.
        self.dirty_counter = 0
        
        #Statistics
        write_header(self.agents_IA, self.agents_FR)
        
        
    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)
        if self.IA_agents_present < self.IA_agents_allowed:
            self.agents[player] = agent_IA(self.IA_network, player, self.nb_rows, self.nb_cols, self.nb_players, self.state_size, self.action_size)
            self.IA_agents_present += 1
            self.agents_IA.append(player)
        else:
            self.agents[player] = agent_FR(self.FR_network, player, self.nb_rows, self.nb_cols, self.nb_players, self.state_size, self.action_size)
            self.agents_FR.append(player)
            
        write_header(self.agents_IA, self.agents_FR)
        

    def register_action(self, row, column, orientation, player):
        """Register action played in game.
        :param row:
        :param columns:
        :param orientation: "v" or "h"
        :param player: 1 or 2
        """
        pass
    
    
    def observe(self, player, players, apples, scores, terminal):
        if self.dirty_counter == 0:
            # Update scores only once per timestep
            self.scores_tMinus1 = self.scores_t
            self.scores_t = scores
            self.rewards = self.scores_t - self.scores_tMinus1
        
        #Push updates
        self.agents[player].observe(self.rewards, players, apples, terminal)
        self.dirty_counter += 1
        
        if self.dirty_counter == len(self.player):
            #All observations have arrived for this timestep. Reset dirty_counter
            self.dirty_counter = 0

    
    def next_action(self, player, players, apples):
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))
        
        action_index = self.agents[player].next_action(players,apples)
        return self.actions[action_index]
        

    def end_game(self):
        if self.ended == False:
            self.ended = True
            
            # store models
            self.IA_network.end_episode()
            self.FR_network.end_episode()
            
            # save scores
            scores_agents_IA = list()
            scores_agents_FR = list()
            for p in self.agents_IA:
                scores_agents_IA.append(self.scores_t[p-1])
            for p in self.agents_FR:
                scores_agents_FR.append(self.scores_t[p-1])   
            write_scores(scores_agents_IA, scores_agents_FR)
            



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
                if msg["nextplayer"] == 1:
                    scores = np.zeros(len(msg["players"]))
                    for j in range(0,len(msg["players"])):
                        scores[j] = msg["players"][j]["score"]
                    games[game].observe(msg["receiver"], msg["players"], msg["apples"], scores, False)
            
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