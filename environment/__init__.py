from myosuite.utils import gym
import os
register = gym.register
model_path = "/home/yuxuan/.local/lib/python3.8/site-packages/myosuite/envs/myo/"
curr_dir = os.path.dirname(os.path.abspath(__file__))
register(
    id = "myoChallengeOslRun-v0",
    entry_point="environment.run_track_v0:RunTrack",
    max_episode_steps=1000,
    kwargs={
            'model_path': model_path+'/assets/leg/myoosl_runtrack.xml',
            'normalize_act': True,
            'reset_type': 'random',  # none, init, random, osl_init
            'terrain': 'flat',  # flat, random, random_mixed
            'hills_difficulties': (0.0, 0.1, 0.0, 0.5, 0.0, 0.8, 0.0, 1.0),
            'rough_difficulties': (0.0, 0.1, 0.0, 0.15, 0.0, 0.2, 0.0, 0.3),
            'stairs_difficulties': (0.0, 0.05, 0.0, 0.1, 0.0, 0.2, 0.0, 0.3),
            'end_pos': -15,
            'frame_skip': 5,
            'start_pos': 14,
            'init_pose_path': model_path+'/assets/leg/sample_gait_cycle.csv',
            'max_episode_steps': 1000
        }
)
register(
    id = "myoChallengeOslRun-v1",
    entry_point="environment.run_track_v1:RunTrack",
    max_episode_steps=1000,
    kwargs={
            'model_path': model_path+'/assets/leg/myoosl_runtrack2.xml',
            'normalize_act': True,
            'reset_type': 'random',  # none, init, random, osl_init
            'terrain': 'flat',  # flat, random, random_mixed
            'hills_difficulties': (0.0, 0.1, 0.0, 0.5, 0.0, 0.8, 0.0, 1.0),
            'rough_difficulties': (0.0, 0.1, 0.0, 0.15, 0.0, 0.2, 0.0, 0.3),
            'stairs_difficulties': (0.0, 0.05, 0.0, 0.1, 0.0, 0.2, 0.0, 0.3),
            'end_pos': -15,
            'frame_skip': 5,
            'start_pos': 14,
            'init_pose_path': model_path+'/assets/leg/sample_gait_cycle.csv',
            'max_episode_steps': 1000
        }
)