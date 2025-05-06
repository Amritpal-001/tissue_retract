import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH

from surrol.utils.pybullet_utils import (
    step,
    render_image,
)

import matplotlib.pyplot as plt

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


CAMERA_MODE_LIST = ["Normal" , "left_far" , "left_closeup"]
CAMERA_MODE = CAMERA_MODE_LIST[0]
DURATION = 0.10 # important for mini-steps

class TissueRetract(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        super(TissueRetract, self)._env_setup(
            # cam_details = { "CAM_YAW": 88.0,
            #                             "CAM_PITCH" : -28.0,
            #                             "CAM_DIST" : 0.78}
            # cam_details = { "CAM_YAW": 90.0,
            #                             "CAM_PITCH" : -30.0,
            #                             "CAM_DIST" : 0.90}
                                        )
        
        self._duration = DURATION # important for mini-steps
        
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?
        # self._duration = 0.05

        self.set_camera_view(mode=CAMERA_MODE)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False


        # tray pad
        tray_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING*2)
        self.obj_ids['fixed'].append(tray_id)  # 1
        # p.changeVisualShape(tray_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))
        
        
        # for x in range(1,100):
        mesh_location = (  (np.random.uniform(low=250, high=275, size=(1))/100)[0] ,
                        (np.random.uniform(low=-20, high=20, size=(1))/100)[0] ,
                        workspace_limits[2][0] - 0.01)
        # print(mesh_location)
        
        self.mesh_location = mesh_location

        ofset = (np.random.rand() - 0.5) * 0.3
        object_location = [mesh_location[0] +0.25, mesh_location[1],  mesh_location[2]+0.02]    
        hidden_tissue_loc = [ mesh_location[0]  + 0.07,
                              mesh_location[1] + ofset, 
                                mesh_location[2]-0.04]
        
        clothID = p.loadSoftBody( os.path.join(ASSET_DIR_PATH,f"cloth/bl_cloth_15_cuts.obj"), basePosition = mesh_location,
                                 baseOrientation = (1, 0, 0, 1), scale = 0.25, 
                                 mass = 0.7, 
                                 useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, 
                                 springElasticStiffness=30, 
                                 frictionCoeff = .5, useFaceContact=0)
        
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            object_location,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        # single Hidden piece
        # hidden_tissue_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
        #                     hidden_tissue_loc ,
        #                     (0, 0, 0, 1),
        #                     useFixedBase=True,
        #                     globalScaling=self.SCALING*1.5 )
        
        # # 4 Hidden spheres
        num_hidden_object = 4
        for _ in range(num_hidden_object):
            hidden_tissue_loc = [ mesh_location[0]  + 0.15 +  (np.random.rand() - 0.5) * 0.05,
                                mesh_location[1] +  (np.random.rand() - 0.5) * 0.2, 
                                    mesh_location[2]-0.04]

            hidden_tissue_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                hidden_tissue_loc ,
                                (0, 0, 0, 1),
                                useFixedBase=True,
                                globalScaling=8 )
            p.changeVisualShape(hidden_tissue_id, -1, rgbaColor=(1, 0,0,1))
            
        
        p.createSoftBodyAnchor(clothID ,103,obj_id,-1, object_location ) #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,102,obj_id,-1, object_location) #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,119,obj_id,-1, object_location) #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,118,obj_id,-1, object_location) #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,133,obj_id,-1, object_location) #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,134,obj_id,-1, object_location) #for 15 cuts cloth

        p.createSoftBodyAnchor(clothID  ,13,-1,-1)
        p.createSoftBodyAnchor(clothID  ,14,-1,-1)

        p.createSoftBodyAnchor(clothID  ,223,-1,-1)
        p.createSoftBodyAnchor(clothID  ,224,-1,-1)

        p.createSoftBodyAnchor(clothID  ,11,-1,-1)
        p.createSoftBodyAnchor(clothID  ,5,-1,-1)

        p.createSoftBodyAnchor(clothID  ,215,-1,-1)
        p.createSoftBodyAnchor(clothID  ,221,-1,-1)
        
        p.createSoftBodyAnchor(clothID  ,45,-1,-1)
        p.createSoftBodyAnchor(clothID  ,150,-1,-1)

        p.createSoftBodyAnchor(clothID  ,0,-1,-1)
        p.createSoftBodyAnchor(clothID  ,210,-1,-1)


        # Textured version
        texture1_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/RustPlain007_COL_VAR1_1K.jpg'))
        texture2_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricLeatherBuffaloRustic001_flat.jpg'))        
        texture3_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricPlainWhiteBlackout009_COL_1K.jpg'))        
        texture4_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricLeatherBuffaloRustic001_flat.jpg'))        

        # Darj red coloured
        # p.changeVisualShape(clothID, -1,  rgbaColor=(1, 0.4,0.6,1) )
        # p.changeVisualShape(tray_id, -1,  rgbaColor=(0.5, 0 ,0.2,1) )

        # Skin coloured version
        # p.changeVisualShape(obj_id, -1, rgbaColor=(1 , 0.5, 0.5, 0.01))
        p.changeVisualShape(clothID, -1,  rgbaColor=(0.96, 0.6 ,0.77,1) )
        p.changeVisualShape(tray_id, -1,  rgbaColor=(0.96, 0.87 ,0.77,1) )

        # p.changeVisualShape(clothID, -1,  rgbaColor=(1, 0.4,0.6,0.01) , textureUniqueId=texture3_id)
        # p.changeVisualShape(obj_id, -1,  rgbaColor=(1, 0.4,0.6,0.01) , textureUniqueId=texture3_id)

        # p.changeVisualShape(tray_id, -1,  rgbaColor=(0.5, 0 ,0.2,1), textureUniqueId=tray_exture_id)

        # p.changeVisualShape(clothID, -1,  rgbaColor=(1, 0.4,0.6,1) , textureUniqueId=cloth_texture_id)
        # p.changeVisualShape(tray_id, -1, textureUniqueId=tray_exture_id)
        
        self.debug = False
        if self.debug == True:
            data = p.getMeshData(clothID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            print("--------------")
            print("data=",data)
            print(data[0])
            print(data[1])
            text_uid = []
            for i in range(data[0]):
                pos = data[1][i]
                uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
                text_uid.append(uid)
        
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        self.max_pixels = 750 #count_dict[6]
        self.max_count = 0
    
    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # time0 = time.time()
        self._set_action(action)
        # time1 = time.time()
        # TODO: check the best way to step simulation
        step(self._duration)

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        
        if isinstance(obs, dict):
            reward  = self.compute_reward(obs['achieved_goal'], self.goal, {})
        else:
            reward  = self.compute_reward(obs, self.goal, {})
        
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo

        # print("reward" , reward, "pixel_info" , pixel_info)
        return obs, reward, done, info

    def _disable_collision(self, obj1 , obj2):
        collisionFilterGroup = 0
        collisionFilterMask = 0
        p.setCollisionFilterGroupMask(obj1, -1, collisionFilterGroup, collisionFilterMask)
        p.setCollisionFilterPair(obj1, obj2, -1, -1, 0)

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(TissueRetract, self)._set_action(action)

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None]  # five waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[4] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, 0., -0.5])  # lift up
        
        if self.debug == True:
            for i , points in enumerate(self._waypoints):
                if i == 0:
                    p.addUserDebugLine([0,0,0], self._waypoints[i][0:3])
                else:
                    p.addUserDebugLine(self._waypoints[i-1][0:3], self._waypoints[i][0:3], 
                                    lineColorRGB = np.random.random(size=3),
                                    lineWidth= 4)

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped object to make it stable
        pose = get_link_pose(self.obj_id, self.obj_link1)
        return pose[0][2] > self._waypoint_z_init + 0.0025 * self.SCALING
        # return True  # mimic the dVRL setting

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.6
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                self._waypoints[i] = None
            break
        return action
    

    def test(self, horizon=100):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        steps, done = 0, False
        obs = self.reset()
        # time.sleep(2)

        while not done and steps <= horizon: #  
            action = self.get_oracle_action(obs)
            obs, reward, done, info = self.step(action)
            # print(steps, " -> achieved goal: {}".format(np.round(self.goal - obs['achieved_goal'], 4)) , info, reward, done)
            print(steps, " -> achieved goal: {}".format(obs['achieved_goal']) , info, reward, done)
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            # time.sleep(0.02)
        print('\n -> Done: {}\n'.format(done > 0))



class TissueRetract_v0(TissueRetract):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.mesh_location[0] + 0.22 ,
                         self.mesh_location[1] ,
                         self.mesh_location[2] +0.08])
        return goal.copy()

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        x = self._is_success(achieved_goal, desired_goal).astype(np.float32) -1.
        return x

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


class TissueRetract_v0_eval(TissueRetract_v0):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.mesh_location[0] + 0.22 + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[1] + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[2] +0.08 + (np.random.rand() - 0.5) * 0.1])
        return goal.copy()

class TissueRetract_v1(TissueRetract):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.mesh_location[0] + 0.22 + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[1] + (np.random.rand() - 0.5) * 0.05,
                         self.mesh_location[2] +0.08])
        return goal.copy()
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        x = self._is_success(achieved_goal, desired_goal).astype(np.float32) -1.
        return x

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


class TissueRetract_v1_eval(TissueRetract_v1):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.mesh_location[0] + 0.22 + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[1] + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[2] +0.08])
        return goal.copy()
    

class TissueRetract_v2(TissueRetract):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.obj_location[0] - 0.03 + (np.random.rand() - 0.5) * 0.1,
                         self.obj_location[1] + (np.random.rand() - 0.5) * 0.05,
                         self.obj_location[2] +0.06])
        return goal.copy()

    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        x = self._is_success(achieved_goal, desired_goal).astype(np.float32) -1.
        return x

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)


    def _env_setup(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        super(TissueRetract, self)._env_setup()

        self._duration = DURATION # important for mini-steps

        self.has_object = True
        self._waypoint_goal = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False


        # tray pad
        tray_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING*2)
        self.obj_ids['fixed'].append(tray_id)  # 1
        # p.changeVisualShape(tray_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))
        
        
        # for x in range(1,100):
        mesh_location = (  (np.random.uniform(low=250, high=275, size=(1))/100)[0] ,
                        (np.random.uniform(low=-20, high=20, size=(1))/100)[0] ,
                        workspace_limits[2][0] - 0.01)

        self.mesh_location = mesh_location

        ofset = (np.random.rand() - 0.5) * 0.3
        object_location2 = [mesh_location[0] +0.25, mesh_location[1],  mesh_location[2]+0.02]    
        object_location1 = [mesh_location[0] +0.25, mesh_location[1]-0.1,  mesh_location[2]+0.02] 
        object_location3 = [mesh_location[0] +0.25, mesh_location[1]+0.1,  mesh_location[2]+0.02] 

        hidden_tissue_loc = [ mesh_location[0]  + 0.07,
                              mesh_location[1] + ofset, 
                                mesh_location[2]-0.04]
        
        clothID = p.loadSoftBody( os.path.join(ASSET_DIR_PATH,f"cloth/bl_cloth_15_cuts.obj"), basePosition = mesh_location,
                                 baseOrientation = (1, 0, 0, 1), scale = 0.25, 
                                 mass = 0.7, 
                                 useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, 
                                 springElasticStiffness=30, 
                                 frictionCoeff = .5, useFaceContact=0)
        
        self.obj_id2 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            object_location2,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        
        self.obj_id1 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                             object_location1  ,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        
        self.obj_id3 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            object_location3   ,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        
        obj_id , self.obj_location = [[self.obj_id1, object_location1],
                            [self.obj_id2, object_location2],
                            [self.obj_id3, object_location3]][np.random.choice([ 0,1,2] )]
        
        # 4 Hidden spheres
        num_hidden_object = 4
        for _ in range(num_hidden_object):
            hidden_tissue_loc = [ mesh_location[0]  + 0.15 +  (np.random.rand() - 0.5) * 0.05,
                                mesh_location[1] +  (np.random.rand() - 0.5) * 0.2, 
                                    mesh_location[2]-0.04]

            hidden_tissue_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                hidden_tissue_loc ,
                                (0, 0, 0, 1),
                                useFixedBase=True,
                                globalScaling=8 )
            p.changeVisualShape(hidden_tissue_id, -1, rgbaColor=(1, 0,0,1))
            
        #for 15 cuts cloth
        p.createSoftBodyAnchor(clothID ,103,self.obj_id2,-1, object_location2 ) 
        p.createSoftBodyAnchor(clothID ,104,self.obj_id2,-1, object_location2) 
        p.createSoftBodyAnchor(clothID ,118,self.obj_id2,-1, object_location2) 
        p.createSoftBodyAnchor(clothID ,119,self.obj_id2,-1, object_location2) 
        p.createSoftBodyAnchor(clothID ,133,self.obj_id2,-1, object_location2) 
        p.createSoftBodyAnchor(clothID ,134,self.obj_id2,-1, object_location2) 

        p.createSoftBodyAnchor(clothID ,43,self.obj_id1,-1, object_location1) 
        p.createSoftBodyAnchor(clothID ,44,self.obj_id1,-1, object_location1) 
        p.createSoftBodyAnchor(clothID ,58,self.obj_id1,-1, object_location1 ) 
        p.createSoftBodyAnchor(clothID ,59,self.obj_id1,-1, object_location1) 
        p.createSoftBodyAnchor(clothID ,73,self.obj_id1,-1, object_location1) 
        p.createSoftBodyAnchor(clothID ,74,self.obj_id1,-1, object_location1) 
        
        p.createSoftBodyAnchor(clothID ,148,self.obj_id3,-1, object_location3) 
        p.createSoftBodyAnchor(clothID ,149,self.obj_id3,-1, object_location3) 
        p.createSoftBodyAnchor(clothID ,163,self.obj_id3,-1, object_location3) 
        p.createSoftBodyAnchor(clothID ,164,self.obj_id3,-1, object_location3) 
        p.createSoftBodyAnchor(clothID ,178,self.obj_id3,-1, object_location3) 
        p.createSoftBodyAnchor(clothID ,179,self.obj_id3,-1, object_location3) 

        p.createSoftBodyAnchor(clothID  ,13,-1,-1)
        p.createSoftBodyAnchor(clothID  ,14,-1,-1)

        p.createSoftBodyAnchor(clothID  ,223,-1,-1)
        p.createSoftBodyAnchor(clothID  ,224,-1,-1)

        p.createSoftBodyAnchor(clothID  ,11,-1,-1)
        p.createSoftBodyAnchor(clothID  ,5,-1,-1)

        p.createSoftBodyAnchor(clothID  ,215,-1,-1)
        p.createSoftBodyAnchor(clothID  ,221,-1,-1)
        
        p.createSoftBodyAnchor(clothID  ,45,-1,-1)
        p.createSoftBodyAnchor(clothID  ,150,-1,-1)

        p.createSoftBodyAnchor(clothID  ,0,-1,-1)
        p.createSoftBodyAnchor(clothID  ,210,-1,-1)

        p.changeVisualShape(tray_id, -1,  rgbaColor=(0.96, 0.87 ,0.77,1) )
        # p.changeVisualShape(self.obj_id1, -1, rgbaColor=(1 , 0.5, 0.5, 0.01))
        # p.changeVisualShape(self.obj_id2, -1, rgbaColor=(1 , 0.5, 0.5, 0.01))
        # p.changeVisualShape(self.obj_id3, -1, rgbaColor=(1 , 0.5, 0.5, 0.01))
        p.changeVisualShape(clothID, -1,  rgbaColor=(0.96, 0.6 ,0.77,1) )

        # Textured version
        # texture1_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/RustPlain007_COL_VAR1_1K.jpg'))
        # texture2_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricLeatherBuffaloRustic001_flat.jpg'))        
        # texture3_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricPlainWhiteBlackout009_COL_1K.jpg'))        
        # texture4_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'textures/FabricLeatherBuffaloRustic001_flat.jpg'))        
        # p.changeVisualShape(self.obj_id1, -1,  rgbaColor=(1, 0.4,0.6,1) , textureUniqueId=texture3_id)
        # p.changeVisualShape(self.obj_id2, -1,  rgbaColor=(1, 0.4,0.6,1) , textureUniqueId=texture3_id)
        # p.changeVisualShape(self.obj_id3, -1,  rgbaColor=(1, 0.4,0.6,1) , textureUniqueId=texture3_id)
        # p.changeVisualShape(clothID, -1,  rgbaColor=(1, 0.4,0.6,1) , textureUniqueId=texture3_id)

        
        self.debug = False
        if self.debug == True:
            data = p.getMeshData(clothID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            print("--------------")
            print("data=",data)
            print(data[0])
            print(data[1])
            text_uid = []
            for i in range(data[0]):
                pos = data[1][i]
                uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1], textSize=0.8)
                text_uid.append(uid)
        
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_ids['rigid'].append(self.obj_id1)  # 0
        self.obj_ids['rigid'].append(self.obj_id2)  # 0
        self.obj_ids['rigid'].append(self.obj_id3)  # 0

        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        self.set_camera_view(mode=CAMERA_MODE)

        self.max_pixels = 750 #count_dict[6]
        self.max_count = 0

class TissueRetract_v3(TissueRetract):
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([self.mesh_location[0] + 0.22 + (np.random.rand() - 0.5) * 0.1,
                         self.mesh_location[1] + (np.random.rand() - 0.5) * 0.05,
                         self.mesh_location[2] +0.08])
        return goal.copy()
    

    '''
        def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        init_img = self.render(mode="numpy")
        score , pixel_info = self.get_retraction_level(init_img[1])
        # plt.imshow(init_img[1])
        # plt.savefig( f"{time.time()}_{np.random.rand()}.png" )
        return score , pixel_info'''

    def get_exposed_tissue_pixel_count(self, segment_img):
        unique_numbers, pixel_counts = np.unique(segment_img, return_counts=True)    
        tissue_pixel_counts = 0 
        for x in zip(unique_numbers, pixel_counts):
            if x[0] >= 6:
                tissue_pixel_counts += x[1]
        return tissue_pixel_counts

    def get_retraction_level(self, segment_img):
        # unique_numbers, pixel_counts = np.unique(segment_img, return_counts=True)    
        # print(unique_numbers  , pixel_counts)
        tissue_pixel_counts = self.get_exposed_tissue_pixel_count(segment_img)
        if tissue_pixel_counts >= 0:   
            return tissue_pixel_counts/self.max_pixels , () #(count_dict[6] , self.max_pixels , self.max_count)
        else:
            return 0 , ()

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        # TODO: may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos = np.array(pos)
            # rotations
            waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
            # relative position state
            object_rel_pos = object_pos - robot_state[0: 3]
        else:
            # TODO: can have a same-length state representation
            object_pos = waypoint_pos = waypoint_rot = object_rel_pos = np.zeros(0)

        if self.has_object:
            # object/waypoint position
            achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos.copy()
        else:
            # tip position
            achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        
        init_img = self.render(mode="numpy")
        score , pixel_info = self.get_retraction_level(init_img[1])
        # print(score, pixel_info)

        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel(),  np.array([score]) # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs
    
    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        # TODO: check the best way to step simulation
        step(self._duration)

        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success({"obs": obs}) , #obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        
        if isinstance(obs, dict):
            reward  = self.compute_reward(obs['achieved_goal'], self.goal, {"obs": obs})
        else:
            reward  = self.compute_reward(obs, self.goal, {"obs": obs})
        
        return obs, reward, done, info


    # def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
    #     """ All sparse reward.
    #     The reward is 0 or -1.
    #     """
    #     # d = goal_distance(achieved_goal, desired_goal)
    #     # return - (d > self.distance_threshold).astype(np.float32)
    #     # print(info)

    #     if isinstance(info, dict):
    #         x = self._is_success(info['obs']).astype(np.float32) 
    #     elif isinstance(info, np.ndarray):
    #         x = np.vectorize(self._is_success)(info)
    #     return x
    
    # def _is_success(self, achieved_goal):
    #     """ Indicates whether or not the achieved goal successfully achieved the desired goal.
    #     """
    #     self.coverage_threshold = 0.8 
    #     d , _ = self.get_retraction_level(achieved_goal)
    #     return np.array(d > self.coverage_threshold) #.astype(np.float32)


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        # d = goal_distance(achieved_goal, desired_goal)
        # return - (d > self.distance_threshold).astype(np.float32)
        # print(info)

        if isinstance(info, dict):
            x = self._is_success(info).astype(np.float32) 
        elif isinstance(info, np.ndarray):
            x = np.vectorize(self._is_success)(info)
        return x
    
    def _is_success(self, info):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        self.coverage_threshold = 0.8 
        return np.array(info['obs']['observation'][-1] > self.coverage_threshold) #.astype(np.float32)




if __name__ == "__main__":
    # from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    env = TissueRetract_v2(render_mode='human')  # create one process and corresponding env

    # envs = [TissueRetract(render_mode='numpy') for x in range(1,5)]

    env.test(horizon=100)
    env.close()
    time.sleep(2)
