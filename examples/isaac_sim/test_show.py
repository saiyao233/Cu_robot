# import torch

# a = torch.zeros(4, device="cuda:0")

# # Standard Library
# import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--headless_mode",
#     type=str,
#     default=None,
#     help="To run headless, use one of [native, websocket], webrtc might not work.",
# )
# parser.add_argument(
#     "--visualize_spheres",
#     action="store_true",
#     help="When True, visualizes robot spheres",
#     default=False,
# )

# parser.add_argument(
#     "--robot", type=str, default="triple_franka.yml", help="robot configuration to load"
# )
# # parser.add_argument(
# #     "--robot", type=str, default="dual_ur10e.yml", help="robot configuration to load"
# # )
# args = parser.parse_args()

# ############################################################

# # Third Party
# from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp(
#     {
#         "headless": args.headless_mode is not None,
#         "width": "1920",
#         "height": "1080",
#     }
# )
# # Third Party
# import carb
# import numpy as np
# from helper import add_extensions, add_robot_to_scene
# from omni.isaac.core import World
# from omni.isaac.core.objects import cuboid, sphere

# ########### OV #################
# from omni.isaac.core.utils.types import ArticulationAction

# # CuRobo
# from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# # from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
# from curobo.geom.sdf.world import CollisionCheckerType
# from curobo.geom.types import WorldConfig
# from curobo.rollout.rollout_base import Goal
# from curobo.types.base import TensorDeviceType
# from curobo.types.math import Pose
# from curobo.types.robot import JointState, RobotConfig
# from curobo.types.state import JointState
# from curobo.util.logger import setup_curobo_logger
# from curobo.util.usd_helper import UsdHelper
# from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# def main():
#     my_world = World(stage_units_in_meters=1.0)
#     stage = my_world.stage

#     xform = stage.DefinePrim("/World", "Xform")
#     stage.SetDefaultPrim(xform)
#     stage.DefinePrim("/curobo", "Xform")
#     # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
#     stage = my_world.stage
#     # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

#     # Make a target to follow

#     # setup_curobo_logger("warn"

#     # warmup curobo instance
#     usd_help = UsdHelper()


#     robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

#     j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
#     default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

#     robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

#     # articulation_controller = robot.get_articulation_controller()

#     world_cfg_table = WorldConfig.from_dict(
#         load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
#     )
#     world_cfg_table.cuboid[0].pose[2] -= 0.02

#     world_cfg1 = WorldConfig.from_dict(
#         load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
#     ).get_mesh_world()
#     world_cfg1.mesh[0].name += "_mesh"
#     world_cfg1.mesh[0].pose[2] = -10.5

#     world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
#     usd_help.load_stage(my_world.stage)
#     usd_help.add_world_to_stage(world_cfg, base_frame="/World")
#     my_world.scene.add_default_ground_plane()
#     while simulation_app.is_running():

#         my_world.step(render=True)
#         # print("x")
#         if not my_world.is_playing():
#             # if i % 100 == 0:
#             #     print("**** Click Play to start simulation *****")
#             # i += 1
#             # if step_index == 0:
#             my_world.play()
    # simulation_app.close()
# import numpy as np 
# def num():
#     x=np.random.normal(0,1,1)
#     y=np.random.normal(2,1,1)
#     z=np.append(x,y)
#     print(z)
# if __name__ == "__main__":
#     # main()
#     num()
# class Parrot(object):
#     def __init__(self):
#         self._voltage = 100000
 
#     @property
#     def voltage(self):
#         """Get the current voltage."""
#         return self._voltage
# c=Parrot()
# print(c.voltage)
# print(c._voltage)
a=0
b=a
a=a+1
print(a,b)
