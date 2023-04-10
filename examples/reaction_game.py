""" reaction_game.py

Reaction time game.
"""
import handyvision.rtgame as rtg

if __name__=="__main__":
    game = rtg.Game(asset_folder = "C:/dev/HandyVision/assets", camera_idx = 0)
    game.run()
