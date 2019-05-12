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
import myLibrary as ml

from agent_IA import agent_IA
from agent_FR import agent_FR
from Keras_DDQNAgent import Keras_DDQNAgent

logger = logging.getLogger(__name__)
games = {}
stats_file = None
agentclass = None
agentcontroller = None
agent_type = None
play_mode = None
model = None

#Statistics
header_written = False
def write_header(agents):
    if header_written == False:
        with open(stats_file, 'w',newline='') as wf:
            writer = csv.writer(wf)
            l1 = [agent_type for _ in agents]
            l2 = ["","TOTAL (SUM)", "MEAN", "STAND. DEVI.", "INEQUALITY", "SUSTAINABILITY", "FIRED", "TIMESTEPS TO GAME END"]
            l = l1 + l2
            writer.writerow(l)
        wf.close()

def set_headers_written():
    global header_written
    header_written = True

def write_scores(scores_agents, sustainability, fired, timestep):
    #never modify header again
    set_headers_written()
    total = -1
    mean = -1
    inequality = -1
    if len(scores_agents) > 0:
        total = np.sum(scores_agents)
        mean = np.mean(scores_agents)
        std = np.std(scores_agents)
        inequality = ml.gini(np.asarray(scores_agents))
    row = scores_agents + ["", total, mean, std, inequality, sustainability, fired, timestep]
    with open(stats_file, 'a',newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(row)
    wf.close()  


"""
Controller of all agents of agent_type

"""
class Agent_Controller:
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
        
        self.player_list = [player]
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        self.state_size = 15*15
        
        #define the mapping of the action scalar to a certain action
        #self.actions = ["left","move","right"]
        self.actions = ["left","move","right", "fire"]
        self.action_size = len(self.actions)
        
        #DDQN
        model_name = model + agent_type
        #model_name = model
        self.network = Keras_DDQNAgent(state_size = self.state_size, action_size = self.action_size, model_name=model_name)

        #Keep one agent per player
        self.agents = dict()
        self.agents[player] = agentclass(self.network, player, self.nb_rows, self.nb_cols, self.nb_players, self.state_size, self.action_size, play_mode)


        
        #Data structures for rewards calculations
        #rewards = scores_t - scores_tMinus1
        self.scores_tMinus1 = np.zeros(nb_players)
        self.scores_t = np.zeros(nb_players)
        self.rewards = np.zeros(nb_players)
        # To shift scores once per timestep (= every player moved), use a dirty_counter
        # This dirty_counter makes sure that every player received an update with the current scores
        # before updating the scores once more.
        self.dirty_counter = 0
        self.timestep = 0
        
        #Statistics
        self.sustainability = np.zeros(nb_players)
        self.nb_rewarded = np.zeros(nb_players)
        self.fired = np.zeros(nb_players)
        write_header(self.player_list)
        
        
    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player_list.append(player)
        self.agents[player] = agentclass(self.network, player, self.nb_rows, self.nb_cols, self.nb_players, self.state_size, self.action_size, play_mode)
        
        #update header for this session
        write_header(self.player_list)
        

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
            self.timestep += 1
            self.scores_tMinus1 = self.scores_t
            self.scores_t = scores
            self.rewards = self.scores_t - self.scores_tMinus1
            #statistics
            self.sustainability += self.rewards * self.timestep
            self.nb_rewarded += self.rewards
        
        #Push updates
        self.agents[player].observe(self.rewards, players, apples, terminal)
        self.dirty_counter += 1
        
        if self.dirty_counter == len(self.player_list):
            #All observations have arrived for this timestep. Reset dirty_counter
            self.dirty_counter = 0

    
    def next_action(self, player, players, apples):
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player_list))
        
        action_index = self.agents[player].next_action(players,apples)
        
        #statistics
        if self.actions[action_index] == "fire":
            self.fired +=1
        
        return self.actions[action_index]
        

    def end_game(self):
        if self.ended == False:
            self.ended = True
            
            # store models
            self.network.end_episode()
            
            # save scores
            scores_agents = list()
            for p in self.player_list:
                scores_agents.append(self.scores_t[p-1])
                
            #compute average sustainability & fired
            susTotal = self.sustainability/(self.nb_rewarded+0.001) #can't be zero
            avgSustainability = np.mean(susTotal)
            avgFired = np.mean(self.fired)
            
            write_scores(scores_agents, avgSustainability, avgFired, self.timestep)
            print("Agent Controller finished.\n\n")
            



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
                    games[msg["game"]] = agentcontroller(msg["player"],
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
            
                if msg["nextplayer"] in games[game].player_list and msg["nextplayer"] == msg["receiver"]:
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
                #print(answer)
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
    global agentcontroller
    global agentclass
    global agent_type
    global play_mode
    global stats_file
    global model
    parser = argparse.ArgumentParser(description='Start agent to play the Apples game')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    parser.add_argument('--type','-t', default="IA", help='agent type is either \'FR\' (Free Rider) or \'IA\' (Inequity Averse, default)')
    parser.add_argument('--play','-p', action='count', default=0, help='In play-mode, the models will not train while playing. As FR and IA use the same model, play-mode is recommended when combining both types of players in a single game.')
    parser.add_argument('--model', '-m', default = "models/DDQN_MODEL_4actions_", type = str, help = 'name of model to store in models folder')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentcontroller = Agent_Controller
    
    model = args.model
    
    if args.type == "FR":
        agent_type = "FR"
        agentclass = agent_FR
    else:
        agent_type = "IA"
        agentclass = agent_IA
    
    if args.play == 0:
        play_mode = False
    else:
        play_mode = True
    
    #Init statistics file
    now = datetime.datetime.now()
    stats_file = "statistics/stats_" + agent_type + "_session_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
