""" game.py

Reaction time game using HandyVision hand psoe estimation.
"""
import handyvision.rtgame as rtg

if __name__=="__main__":
    game = rtg.Game(asset_folder = "assets", camera_idx = 0)
    game.run()
