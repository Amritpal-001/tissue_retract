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
        super(TissueRetract, self)._env_setup(cam_details = { "CAM_YAW": 88.0,
                                        "CAM_PITCH" : -28.0,
                                        "CAM_DIST" : 0.78})
        
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?
        # self._duration = 0.4

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
        p.changeVisualShape(tray_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))
        
            
        cloth_type = 15
        anchor_type = ['center' , 'left_center' , 'top_center' , 'right_center' , 'bottom_center'][4]

        mesh_location = (workspace_limits[0].mean() -0.1,  # TODO: scaling
                    workspace_limits[1].mean() -0.1,
                    workspace_limits[2][0] - 0.01)

        ofset = (np.random.rand() - 0.5) * 0.3
        object_location = [mesh_location[0] +0.25, mesh_location[1]+0.04,  mesh_location[2]+0.02]    
        hidden_tissue_loc = [mesh_location[0] +0.10 , mesh_location[1] + ofset,  mesh_location[2]-0.041]
        
         
        clothID = p.loadSoftBody( os.path.join(ASSET_DIR_PATH,f"cloth/bl_cloth_{cloth_type}_cuts.obj"), basePosition = mesh_location,
                                 baseOrientation = (1, 0, 0, 1), scale = 0.25, 
                                 mass = 0.7, 
                                 useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, 
                                 springElasticStiffness=10, 
                                #  springDampingStiffness=.1, 
                                #  springDampingAllDirections = 1, useSelfCollision = 0, 
                                 frictionCoeff = .5, useFaceContact=0)
        
        obj_id1 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            [mesh_location[0] +0.25, mesh_location[1]-0.1,  mesh_location[2]+0.02]  ,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        
        obj_id2 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                               [mesh_location[0] +0.25, mesh_location[1],  mesh_location[2]+0.02]  ,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        
        obj_id3 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                               [mesh_location[0] +0.25, mesh_location[1]+0.1,  mesh_location[2]+0.02] ,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.5)
        

        obj_id = np.random.choice([obj_id1,obj_id3, obj_id2])
        
        hidden_tissue_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            hidden_tissue_loc ,
                            (0, 0, 0, 1),
                            useFixedBase=True,
                            globalScaling=self.SCALING*2 )
                            
    
        # Anchor object to mesh
        if cloth_type == 15 and anchor_type=="center":
            p.createSoftBodyAnchor(clothID ,83,obj_id,-1, object_location) #for 10 cuts cloth
        if cloth_type == 15 and anchor_type=="bottom_center":
            p.createSoftBodyAnchor(clothID ,59,obj_id1,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,74,obj_id1,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,89,obj_id1,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,58,obj_id1,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,73,obj_id1,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,88,obj_id1,-1, object_location) #for 15 cuts cloth


            p.createSoftBodyAnchor(clothID ,104,obj_id2,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,119,obj_id2,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,103,obj_id2,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,118,obj_id2,-1, object_location) #for 15 cuts cloth
            
            p.createSoftBodyAnchor(clothID ,149,obj_id3,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,164,obj_id3,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,179,obj_id3,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,148,obj_id3,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,163,obj_id3,-1, object_location) #for 15 cuts cloth
            p.createSoftBodyAnchor(clothID ,178,obj_id3,-1, object_location) #for 15 cuts cloth

            p.createSoftBodyAnchor(clothID  ,29,-1,-1)
            p.createSoftBodyAnchor(clothID  ,14,-1,-1)

            p.createSoftBodyAnchor(clothID  ,209,-1,-1)
            p.createSoftBodyAnchor(clothID  ,224,-1,-1)

            p.createSoftBodyAnchor(clothID  ,11,-1,-1)
            p.createSoftBodyAnchor(clothID  ,5,-1,-1)

            p.createSoftBodyAnchor(clothID  ,215,-1,-1)
            p.createSoftBodyAnchor(clothID  ,221,-1,-1)
            
            p.createSoftBodyAnchor(clothID  ,45,-1,-1)
            p.createSoftBodyAnchor(clothID  ,150,-1,-1)

            p.createSoftBodyAnchor(clothID  ,1,-1,-1)
            p.createSoftBodyAnchor(clothID  ,210,-1,-1)

        elif cloth_type == 10:
            p.createSoftBodyAnchor(clothID ,65,obj_id,-1, object_location) #for 10 cuts cloth
        else: # cloth_type == 5:
            p.createSoftBodyAnchor(clothID ,12,obj_id,-1, object_location) #for 15 cuts cloth

        p.changeVisualShape(obj_id, -1, rgbaColor=(225, 225 / 255, 225 / 255))
        p.changeVisualShape(clothID, -1, rgbaColor=(225, 225 / 255, 225 / 255))
        
        debug = False
        if debug:
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


        # render related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=0.78 * self.SCALING,
            yaw=88,
            pitch=-28,
            roll=0,
            upAxisIndex=2
        )

        # Initial pixel count 
        # init_img = self.render(mode="numpy")
        # unique_numbers, pixel_counts = np.unique(init_img[1], return_counts=True)  
        # count_dict = dict(zip(unique_numbers, pixel_counts))
        # print(count_dict)
        self.max_pixels = 20000 #count_dict[6]
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
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        # d = goal_distance(achieved_goal, desired_goal)
        # return - (d > self.distance_threshold).astype(np.float32)

        x = self._is_success(achieved_goal, desired_goal).astype(np.float32) -1.
        # print(x , achieved_goal.shape, desired_goal.shape)
        return x
    

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def get_retraction_level(self, segment_img):
        unique_numbers, pixel_counts = np.unique(segment_img, return_counts=True)    
        # print(unique_numbers  , pixel_counts)
        if 6 in unique_numbers:   
            count_dict = dict(zip(unique_numbers, pixel_counts))
            if count_dict[6] > self.max_count:
                self.max_count = count_dict[6]
            curr_count = 1 - ( self.max_pixels - count_dict[6])/ self.max_pixels
            return curr_count , (count_dict[6] , self.max_pixels , self.max_count)
        else:
            return 0 , ()

    def _disable_collision(self, obj1 , obj2):
        collisionFilterGroup = 0
        collisionFilterMask = 0
        p.setCollisionFilterGroupMask(obj1, -1, collisionFilterGroup, collisionFilterMask)
        p.setCollisionFilterPair(obj1, obj2, -1, -1, 0)

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(TissueRetract, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.1 ,
                         workspace_limits[1].mean() -0.06 + (np.random.rand() - 0.5) * 0.1,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

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

        while not done and  steps <= horizon:
            action = self.get_oracle_action(obs)
            obs, reward, done, info = self.step(action)
            print(steps, " -> achieved goal: {}".format(np.round(self.goal - obs['achieved_goal'], 4)) , info, reward, done)
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))



if __name__ == "__main__":
    env = TissueRetract(render_mode='human')  # create one process and corresponding env

    env.test(horizon=100)
    env.close()
    time.sleep(2)
