"""
Created on mars. 26, 2024
@author: Pierre Maillard
"""

from collections import defaultdict
import numpy as np

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from coomm.callback_func import RodCallBack, CylinderCallBack

class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass

class RibbonEnvironment:
    def __init__(self, final_time, time_step=1.0e-6, recording_fps=30):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time/self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (self.recording_fps * self.time_step))

    def get_systems(self,):
        return self.simulator
    
    def get_data(self,):
        return [self.rod_parameters_dict]

    def set_ribbon(self):
        base_length, radius = self.set_rod()

    def setup(self):        
        self.set_ribbon()

    def set_rod(self):
        """ Set up a rod """
        n_elements = 300        # number of discretized elements of the rod
        base_length = 0.5       # total length of the rod
        radius_base = 0.05     # radius of the rod
        
        self.shearable_rod = CosseratRod.straight_rod(
            n_elements=n_elements,
            start=np.zeros((3,)),
            direction=direction,
            normal=np.array([0.0, 0.0, -1.0]),
            base_length=base_length,
            base_radius=radius,
            density=1050,
            youngs_modulus=10_000,
            shear_modulus=10_000 / (2*(1 + 0.5)),
        )

        self.simulator.append(self.shearable_rod)

        damping_constant = 0.05
        self.simulator.dampen(shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=dt,
        )
        
        self.rod_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            RodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict
        )

        """ Set up boundary conditions """
        
        self.simulator.constrain(self.shearable_rod).using(
            GeneralConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
            translational_constraint_selector=np.array([False, True, True]),  # Allow X movement, fix Y & Z
            rotational_constraint_selector=np.array([False, True, True])  # Fix all rotations
        )
        
        
        sef.simulator.constrain(self.shearable_rod).using(
            GeneralConstraint,
            constrained_position_idx=(-1,),
            constrained_director_idx=(-1,),
            rod_frame_bool = True,
            translational_constraint_selector=np.array([False, False, True]),  
            rotational_constraint_selector=np.array([False, False, False]) 
        )

        
        self.simulator.add_forcing_to(shearable_rod).using(
            EndpointTorque, origin_torque, end_torque, ramp_up_time=ramp_up_time_torque
        )
        
        
        self.simulator.add_forcing_to(shearable_rod).using(
            EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time_force
        )

        return base_length, radius


    def set_BC_control(self,
        base_length, base_radius, tip_radius,
        arm, arm_parameters_dict
    ):
        """ Add drag force """
        dl = base_length/arm.n_elems
        fluid_factor = 1
        r_bar = (base_radius + tip_radius) / 2
        sea_water_dentsity = 1022
        c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
        c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor
        
        self.simulator.add_forcing_to(arm).using(
            DragForce,
            rho_environment=sea_water_dentsity,
            c_per=c_per,
            c_tan=c_tan,
            system=arm,
            step_skip=self.step_skip,
            callback_params=arm_parameters_dict # self.rod_parameters_dict
        )

    def reset(self):
        self.simulator = BaseSimulator()

        self.setup()  

        """ Finalize the simulator and create time stepper """
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        """ Return 
            (1) total time steps for the simulation step iterations
            (2) systems for controller design
        """
        return self.total_steps, self.get_systems()

    def step(self, time, muscle_activations):

        """ Set muscle activations """
        for muscle_group, activation in zip(self.muscle_groups, muscle_activations):
            muscle_group.apply_activation(activation)
        
        """ Run the simulation for one step """
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print("NaN detected in the simulation !!!!!!!!")
            done = True

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done

    def save_data(self, filename="simulation", **kwargs):
        
        import pickle

        print("Saving data to pickle files ...", end='\r')

        with open(filename+"_data.pickle", "wb") as data_file:
            data = dict(
                recording_fps=self.recording_fps,
                systems=self.get_data(),
                muscle_groups=self.muscle_callback_params_list,
                **kwargs
            )
            pickle.dump(data, data_file)

        with open(filename+"_systems.pickle", "wb") as system_file:
            data = dict(
                systems=self.get_systems(),
                muscle_groups=self.muscle_groups,
            )
            pickle.dump(data, system_file)

        print("Saving data to pickle files ... Done!")
