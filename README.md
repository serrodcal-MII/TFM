# TFM

This project uses `ffmpeg`, so install it with Homebrew as follows:

```shell script
brew install ffmpeg  
```

Next, install all the requirementes:

```shell script
python -m pip install -r requirements.txt 
```

Then, install Atari games:

```shell script
python -m pip install gymnasium\[atari\]
```

Finally, install `accept-rom-license`:

```shell script
python -m pip install gymnasium\[accept-rom-license\]
```

If you get the error below, please use AutoROM (included in `requirements.txt`) to fix it:

```
A.L.E: Arcade Learning Environment (version 0.8.1+53f58b7)
[Powered by Stella]
Traceback (most recent call last):
  File "/Users/sergiorodriguezcalvo/Repositories/Miscellaneous/TFM/reproduce/src/atari/breakout.py", line 42, in <module>
    atari = gym.make("ALE/Breakout-v5")
  File "/Users/sergiorodriguezcalvo/Repositories/Miscellaneous/TFM/reproduce/.venv/lib/python3.10/site-packages/gymnasium/envs/registration.py", line 642, in make
    env = env_creator(**_kwargs)
  File "/Users/sergiorodriguezcalvo/Repositories/Miscellaneous/TFM/reproduce/.venv/lib/python3.10/site-packages/shimmy/atari_env.py", line 171, in __init__
    self.seed()
  File "/Users/sergiorodriguezcalvo/Repositories/Miscellaneous/TFM/reproduce/.venv/lib/python3.10/site-packages/shimmy/atari_env.py", line 216, in seed
    raise Error(
gymnasium.error.Error: We're Unable to find the game "Breakout". Note: Gymnasium no longer distributes ROMs. If you own a license to use the necessary ROMs for research purposes you can download them via `pip install gymnasium[accept-rom-license]`. Otherwise, you should try importing "Breakout" via the command `ale-import-roms`. If you believe this is a mistake perhaps your copy of "Breakout" is unsupported. To check if this is the case try providing the environment variable `PYTHONWARNINGS=default::ImportWarning:ale_py.roms`. For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management
```

```shell script
AutoROM --accept-license
```