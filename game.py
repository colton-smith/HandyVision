""" game.py

Reaction time game using HandyVision hand pose estimation.

To run me:

pip install -r requirements.txt
pip install -e lib
python game.py

Note: Use a virtual environment if you don't want to 
muddle your global install!
"""
import handyvision.rtgame as rtg

if __name__=="__main__":
    game = rtg.Game(asset_folder = "assets", camera_idx = 0)
    game.run()
