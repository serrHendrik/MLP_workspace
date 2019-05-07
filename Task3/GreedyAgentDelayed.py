#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2018-2019)
of Karl Tuys and Wannes Meert.
Copyright (c) 2019 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio

import numpy as np
import websockets
import json
from collections import defaultdict
import random

logger = logging.getLogger(__name__)
games = {}
agentclass = None


class GreedyAgent:
    """Example agent implementation base class.
    It moves the agent one position forward.
    A Agent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game
    This class does not necessarily use the best data structures for the
    approach you want to use.
    """
    def __init__(self, player, nb_rows, nb_cols,nb_players):
        """Create an agent.
        :param player: Player number
        :param nb_rows: Rows in grid
        :param nb_cols: Columns in grid
        """
        self.player = {player}
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.nb_players = nb_players
        self.turn = 0

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)

    def register_action(self, row, column, orientation, player):
        """Register action played in game.
        :param row:
        :param columns:
        :param orientation: "left", "right", "up" or "down"
        :param player id
        """
        pass

    def next_action(self,  player, players, apples):
        if player == 1:
            self.turn+=1
        """Return the next action this agent wants to perform.
        In this example, the function implements a random move. Replace this
        function with your own approach.
        :return: (row, column, orientation)
        """
        logger.info("Computing next move (grid={}x{}, player={})"\
                .format(self.nb_rows, self.nb_cols, self.player))

        me = players[player-1]
        location = np.array(me.get('location'))
        orientation = me.get('orientation')

        # Build a 15x15 matrix representing relative positions of apples to the snake
        # The snake is in the center positon
        # positions with apples have value 1, the rest has value 0
        appleMatrix = np.zeros((15,15))
        for apple in apples:
            a = np.array(apple) - np.array(location) + np.array([7,7])
            a[0]=a[0]%self.nb_cols
            a[1]=a[1]%self.nb_rows
            appleMatrix[a[1],a[0]]=1
        rotate = {
            'left':  np.rot90(appleMatrix, 3),
            'right': np.rot90(appleMatrix, 1),
            'up':    np.rot90(appleMatrix, 0),
            'down':  np.rot90(appleMatrix, 2)
        }
        appleMatrix = rotate[orientation]

        #Number of steps needed to reach a position
        distanceMatrix=np.array([
            (14,13,12,11,10,9, 8, 7, 8, 9, 10,11,12,13,14),
            (13,12,11,10,9, 8, 7, 6, 7, 8, 9, 10,11,12,13),
            (12,11,10,9, 8, 7, 6, 5, 6, 7, 8, 9, 10,11,12),
            (11,10,9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9, 10,11),
            (10,9, 8, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 9, 10),
            (9, 8, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9 ),
            (8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8 ),
            (7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7 ),
            (8, 7, 6, 5, 4, 3, 2, 3, 2, 3, 4, 5, 6, 7, 8 ),
            (9, 8, 7, 6, 5, 4, 3, 4, 3, 4, 5, 6, 7, 8, 9 ),
            (10,9, 8, 7, 6, 5, 4, 5, 4, 5, 6, 7, 8, 9, 10),
            (11,10,9, 8, 7, 6, 5, 6, 5, 6, 7, 8, 9, 10,11),
            (12,11,10,9, 8, 7, 6, 7, 6, 7, 8, 9, 10,11,12),
            (13,12,11,10,9, 8, 7, 8, 7, 8, 9, 10,11,12,13),
            (14,13,12,11,10,9, 8, 9, 8, 9, 10,11,12,13,14)
        ])

        #calculate a score per action by counting how easy it is to reach apples after taking that action
        #A weight matrix is used so both the number of apples and how closer they are matter
        #this way clusters of apples can be preferred over a single closer apple
        weightMatrix = np.exp((15-distanceMatrix))
        nextStepApples={
            "left": np.roll(np.rot90(appleMatrix,3),1,axis=0),
            "right":np.roll(np.rot90(appleMatrix,1),1,axis=0),
            "move": np.roll(         appleMatrix   ,1,axis=0)
        }
        score ={
            "left":np.sum(np.multiply(weightMatrix,nextStepApples["left"])),
            "right":np.sum(np.multiply(weightMatrix,nextStepApples["right"])),
            "move":np.sum(np.multiply(weightMatrix,nextStepApples["move"])),
        }
        #select the move with the highest score
        move = max(score, key=score.get)
        #if no apples are close, run horizontally over the field since discovers the most squares
        if score[move]==0:
            if (orientation=="up" or orientation=="down"):
                move = "left"
            else: move = "move"
        #don't eat apples at the start of the game
        if self.turn<100:
            move = "left"
        return move

    def end_game(self,scores,receiver):
        if receiver ==1:
            print("scores:" + str(scores))
        self.ended = True

    def observe(self, player, players, apples, scores, terminal):
        pass


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
            # print("JSON MSG:\n" + str(msg) + "\n")
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
                    for j in range(0, len(msg["players"])):
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
                games[msg["game"]].end_game(scores, msg["receiver"])
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

    agentclass = GreedyAgent
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())