"""
Created on mars. 26, 2024
@author: Pierre Maillard
"""

from collections import defaultdict
import numpy as np

import elastica as el
from coomm.callback_func import SphereCallBack

from examples.set_arm_environment import ArmEnvironment

class Environment(ArmEnvironment):
    
    def get_data(self):
        return [self.rod_parameters_dict, self.sphere_parameters_dict]

    def setup(self):
        self.set_arm()
        self.set_target()

    def set_target(self):
        """ Set up a sphere object """
        target_radius = 0.006
        self.sphere = el.Sphere(
            center=np.array([0.01, 0.15, 0.06]),
            base_radius=target_radius,
            density=1000
        )
        self.sphere.director_collection[:, :, 0] = np.array(
            [[ 0, 0, 1],
             [ 0, 1, 0],
             [-1, 0, 0]]
        )
        self.simulator.append(self.sphere)
        self.sphere_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.sphere).using(
            SphereCallBack,
            step_skip=self.step_skip,
            callback_params=self.sphere_parameters_dict
        )
        
        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            el.OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,)
        )