/*
 *  `app` module
 *  ============
 *
 *  Provides the game initialization routine.
 */

import 'phaser';
//  Import game instance configuration.
import * as config from '@/config';

//  Boot the game.
export function boot() {
  window.game = new Phaser.Game(config);
  return window.game;
}

boot();
