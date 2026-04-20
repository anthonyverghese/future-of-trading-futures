"""Modular backtest library for MNQ alert/bot trading strategies.

Modules:
  data       — load days, precompute per-tick factor arrays
  zones      — pluggable zone state machines (human, bot)
  scoring    — score functions, weight training, human weight constants
  evaluate   — trade outcome evaluation (target/stop/timeout)
  simulate   — tick-by-tick simulation with integrated scoring + constraints
  replay     — simplified replay for pre-simulated entries
  report     — formatting and summary stats
  walk_forward — generic walk-forward framework
"""
