import collections
from myosuite.utils import gym
import numpy as np
import os
from enum import Enum
from typing import Optional, Tuple
import copy
import csv
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat, quat2euler_intrinsic, intrinsic_euler2quat, quat2mat
from myosuite.envs.heightfields import TrackField
from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController

class TrackTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2
    STAIRS = 3
    MIXED = 4


class WalkEnvV1(BaseV0):
    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'phase_var',
        'muscle_length',
        'muscle_velocity',
        'muscle_force'
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0
    }
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height = 0.8,
               max_rot = 0.8,
               hip_period = 100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        obs_dict['phase_var'] = np.array([(self.steps/self.hip_period) % 1]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict
    
    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r'])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip',  cyclic_hip),
            ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),
            # Must keys
            ('sparse',  vel_reward),
            ('solved',    vel_reward >= 1.0),
            ('done',  self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if  self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs


    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        Get a reward proportional to the specified joint angles.
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_r][2]])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([self.sim.data.body_xpos[foot_id_l]-self.sim.data.body_xpos[pelvis], self.sim.data.body_xpos[foot_id_r]-self.sim.data.body_xpos[pelvis]])

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(-np.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_rot_condition(self):
        """
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]] for name in names])


class RunTrack(WalkEnvV1):
    DEFAULT_OBS_KEYS = [
        'internal_qpos',
        'internal_qvel',
        'grf',
        'torso_angle',
        'model_root_pos',
        'model_root_vel',
        'muscle_length',
        'muscle_velocity',
        'muscle_force',
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 1,
        "solved": +10,
    }
    # OSL-related paramters
    ACTUATOR_PARAM = {}
    OSL_PARAM_LIST = []
    OSL_PARAM_SELECT = 0

    biological_jnt = ['rajlumbar_extension','rajlumbar_bending','rajlumbar_rotation',
                      'rajarm_flex_r','rajarm_add_r','rajarm_rot_r','rajelbow_flex_r',
                      'rajpro_sup_r','rajwrist_flex_r','rajwrist_dev_r','rajarm_flex_l',
                      'rajarm_add_l','rajarm_rot_l','rajelbow_flex_l','rajpro_sup_l',
                      'rajwrist_flex_l','rajwrist_dev_l',
                      'hip_adduction_l', 'hip_flexion_l', 'hip_rotation_l',
                      'hip_adduction_r', 'hip_flexion_r', 'hip_rotation_r', 
                      'knee_angle_l', 'knee_angle_l_beta_rotation1', 
                      'knee_angle_l_beta_translation1', 'knee_angle_l_beta_translation2', 
                      'knee_angle_l_rotation2', 'knee_angle_l_rotation3', 'knee_angle_l_translation1', 
                      'knee_angle_l_translation2', 'mtp_angle_l', 'ankle_angle_l', 
                      'subtalar_angle_l'
                      ]
    upper_biological_jnt = ['rajlumbar_extension','rajlumbar_bending','rajlumbar_rotation',
                      'rajarm_flex_r','rajarm_add_r','rajarm_rot_r','rajelbow_flex_r',
                      'rajpro_sup_r','rajwrist_flex_r','rajwrist_dev_r','rajarm_flex_l',
                      'rajarm_add_l','rajarm_rot_l','rajelbow_flex_l','rajpro_sup_l',
                      'rajwrist_flex_l','rajwrist_dev_l']

    biological_act = ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagIsch_l', 'addmagMid_l', 
                      'addmagProx_l', 'bflh_l', 'bfsh_l', 'edl_l', 'ehl_l', 'fdl_l', 'fhl_l', 'gaslat_l', 'gasmed_l', 
                      'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 'glmed1_r', 
                      'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 
                      'glmin3_l', 'glmin3_r', 'grac_l', 'iliacus_l', 'iliacus_r', 
                      'perbrev_l', 'perlong_l', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'recfem_l', 'sart_l', 
                      'semimem_l', 'semiten_l', 'soleus_l', 'tfl_l', 'tibant_l', 'tibpost_l', 'vasint_l', 
                      'vaslat_l', 'vasmed_l']
    upper_biological_act = ['lumbar_ext', 'lumbar_bend', 'lumbar_rot', 'shoulder_flex_r',
                            'shoulder_add_r', 'shoulder_rot_r', 'elbow_flex_r', 'pro_sup_r',
                            'wrist_flex_r', 'wrist_dev_r', 'shoulder_flex_l', 'shoulder_add_l',
                            'shoulder_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        BaseV0.__init__(self, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)
    def _setup(self, obs_keys: list = DEFAULT_OBS_KEYS, 
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS, 
               reset_type='init',
               terrain='random',
               hills_difficulties=(0,0),
               rough_difficulties=(0,0),
               stairs_difficulties=(0,0),
               real_width=1,
               end_pos = -15,
               start_pos = 14,
               init_pose_path=None,
               osl_param_set=4,
               max_episode_steps=36000,
               **kwargs):
        self.startFlag = False
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.terrain_type = TrackTypes.FLAT.value
        self.osl_param_set = osl_param_set

        
        self.OSL_CTRL = MyoOSLController(np.sum(self.sim.model.body_mass), init_state='e_stance', n_sets=self.osl_param_set)
        self.OSL_CTRL.start()

        self.muscle_space = self.sim.model.na #54
        print("Num of Muscle:", self.muscle_space)
        self.full_ctrl_space = self.sim.model.nu #73
        print("Num of Ctrl:", self.full_ctrl_space)

        self._get_actuator_params() #Init Prosthesis Actuator Parameters
        self._setup_convenience_vars()
        
        self.trackfield = TrackField(
            sim=self.sim,
            rng=self.np_random,
            rough_difficulties=rough_difficulties,
            hills_difficulties=hills_difficulties,
            stairs_difficulties=stairs_difficulties,
            reset_type=terrain,
        )
        self.real_width = real_width
        self.reset_type = reset_type
        self.terrain = terrain
        self.grf_sensor_names = ['l_foot', 'l_toes']
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reset_type=reset_type,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['terrain'] = np.array([self.terrain_type])
        obs_dict['internal_qpos'] = self.get_internal_qpos()
        obs_dict['internal_qvel'] = self.get_internal_qvel() 
        obs_dict['grf'] = self._get_grf().copy()
        obs_dict['socket_force'] = self._get_socket_force().copy()
        obs_dict['torso_angle'] = self.sim.data.body('pelvis').xquat.copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()
        obs_dict['upper_joint_angle'] = self.upper_joint_qpos()
        obs_dict['upper_joint_velocity'] = self.upper_joint_qvel()
        obs_dict['upper_joint_torque'] = self.upper_joint_torque()

        obs_dict['model_root_pos'] = sim.data.qpos[:2].copy()
        obs_dict['model_root_vel'] = sim.data.qvel[:2].copy()

        if not self.trackfield is None:
            obs_dict['hfield'] = self.trackfield.get_heightmap_obs()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pass


    def _get_actuator_params(self):
        actuators = ['osl_knee_torque_actuator', 'osl_ankle_torque_actuator', ]
        for actu in actuators:
            self.ACTUATOR_PARAM[actu] = {}
            self.ACTUATOR_PARAM[actu]['id'] = self.sim.data.actuator(actu).id
            self.ACTUATOR_PARAM[actu]['Fmax'] = np.max(self.sim.model.actuator(actu).ctrlrange) * self.sim.model.actuator(actu).gear[0]
    
    def _setup_convenience_vars(self):
        self.actuator_names = np.array(self._get_actuator_names())# Only Muscle
        self.osl_actuator_names = np.array([['osl_knee_torque_actuator', 'osl_ankle_torque_actuator', ]])
        self.upper_actuator_names = np.array(self._get_upper_actuators_names())
        self.joint_names = np.array(self._get_joint_names())
        self.muscle_fmax = np.array(self._get_muscle_fmax())
        self.muscle_lengthrange = np.array(self._get_muscle_lengthRange())
        self.tendon_len = np.array(self._get_tendon_lengthspring())
        self.musc_operating_len = np.array(self._get_muscle_operating_length())
    
    def _get_upper_actuators_names(self):
        return [self.sim.model.actuator(act_id).name for act_id in range(self.sim.model.na+2, self.sim.model.nu)]

    def _get_actuator_names(self):
        '''
        Return a list of actuator names according to the index ID of the actuators
        '''
        return [self.sim.model.actuator(act_id).name for act_id in range(1, self.sim.model.na)]
    
    def _get_joint_names(self):
        '''
        Return a list of joint names according to the index ID of the joint angles
        '''
        return [self.sim.model.joint(jnt_id).name for jnt_id in range(1, self.sim.model.njnt)]
    
    def _get_muscle_fmax(self):
        return self.sim.model.actuator_gainprm[:self.muscle_space, 2].copy()

    def _get_muscle_lengthRange(self):
        return self.sim.model.actuator_lengthrange[:self.muscle_space].copy()

    def _get_tendon_lengthspring(self):
        return self.sim.model.tendon_lengthspring.copy()

    def _get_muscle_operating_length(self):
        return self.sim.model.actuator_gainprm[:self.muscle_space,0:2].copy()
    
    def get_internal_qpos(self):
        """
        Get the internal joint positions without the osl leg joints.
        """
        temp_qpos = np.zeros(len(self.biological_jnt),)
        counter = 0
        for jnt in self.biological_jnt:
            temp_qpos[counter] = self.sim.data.joint(jnt).qpos[0].copy()
            counter += 1
        return temp_qpos
    
    def get_internal_qvel(self):
        """
        Get the internal joint velocities without the osl leg joints.
        """
        temp_qvel = np.zeros(len(self.biological_jnt),)
        counter = 0
        for jnt in self.biological_jnt:
            temp_qvel[counter] = self.sim.data.joint(jnt).qvel[0].copy()
            counter += 1
        return temp_qvel * self.dt
    
    def _get_grf(self):
        grf = np.array([self.sim.data.sensor(sens_name).data[0] for sens_name in self.grf_sensor_names]).copy()
        return grf
    
    def _get_socket_force(self):
        return self.sim.data.sensor('r_socket_load').data.copy()
    
    def muscle_lengths(self):
        """
        Get the muscle lengths. Remove the osl leg actuators from the data.
        """
        temp_len = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_len[counter] = self.sim.data.actuator(jnt).length[0].copy()
            counter += 1
        return temp_len

    def muscle_forces(self):
        """
        Get the muscle forces. Remove the osl leg actuators from the data.
        """
        temp_frc = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_frc[counter] = self.sim.data.actuator(jnt).force[0].copy()
            counter += 1
        return np.clip(temp_frc / 1000, -100, 100)

    def muscle_velocities(self):
        """
        Get the muscle velocities. Remove the osl leg actuators from the data.
        """
        temp_vel = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_vel[counter] = self.sim.data.actuator(jnt).velocity[0].copy()
            counter += 1
        return np.clip(temp_vel, -100, 100)

    def upper_joint_qpos(self):
        temp_angle = np.zeros(len(self.upper_biological_jnt),)
        counter = 0
        for jnt in self.upper_biological_jnt:
            temp_angle[counter] = self.sim.data.joint(jnt).qpos[0].copy()
            counter += 1
        return temp_angle
    
    def upper_joint_qvel(self):
        temp_qvel = np.zeros(len(self.upper_biological_jnt),)
        counter = 0
        for jnt in self.upper_biological_jnt:
            temp_qvel[counter] = self.sim.data.joint(jnt).qvel[0].copy()
            counter += 1
        return temp_qvel * self.dt
    
    def upper_joint_torque(self):
        temp_torque = np.zeros(len(self.upper_biological_act),)
        counter = 0
        for jnt in self.upper_biological_act:
            temp_torque[counter] = self.sim.data.actuator(jnt).force[0].copy()
            counter += 1
        return np.clip(temp_torque, -150, 150)
    