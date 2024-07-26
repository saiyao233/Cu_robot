import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################


# def draw_points(rollouts: torch.Tensor):
#     if rollouts is None:
#         return
#     # Standard Library
#     import random

#     # Third Party
#     from omni.isaac.debug_draw import _debug_draw

#     draw = _debug_draw.acquire_debug_draw_interface()
#     N = 100
#     # if draw.get_num_points() > 0:
#     draw.clear_points()
#     cpu_rollouts = rollouts.cpu().numpy()
#     b, h, _ = cpu_rollouts.shape
#     point_list = []
#     colors = []
#     for i in range(b):
#         # get list of points:
#         point_list += [
#             (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
#         ]
#         colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
#     sizes = [10.0 for _ in range(b * h)]
#     draw.draw_points(point_list, colors, sizes)
class FrankaRmpFlowExample():
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        #Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rmpflow_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
            end_effector_frame_name = "right_gripper",
            maximum_substep_size = 0.00334
        )

        #Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation,self._rmpflow)

        self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(
            target_position, target_orientation
        )

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be

        self._target.set_world_pose(np.array([.5,0,.7]),euler_angles_to_quats([0,np.pi,0]))

def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    # j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    # robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02
    # robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    # articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    franka = Franka(prim_path="/Franka", name=f"manipulator")
    controller = RMPFlowController(name=f"controller", robot_articulation=franka)
    target_pos = np.array([0.0, 0.0, 1.0])
    target_rot = np.array([1.0, 0.0, 0.0, 0.0]) # wxyz quaternion

    init_world = False
    add_extensions(simulation_app, args.headless_mode)
    step=0
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        # # draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue
        actions = controller.forward(
        target_end_effector_position=target_pos,
        target_end_effector_orientation=target_rot,
)
        # print(actions)
        # franka.apply_action(actions)
        # step_index = my_world.current_time_step_index
        # franka.update(step_index)
        # if step_index <= 2:
        #     my_world.reset()
      
        # if not init_curobo:
        #     init_curobo = True
        step += 1
        # step_index = step
        # if step_index % 1000 == 0:
        #     print("Updating world")
        #     obstacles = usd_help.get_obstacles_from_stage(
        #         # only_paths=[obstacles_path],
        #         ignore_substring=[
        #             robot_prim_path,
        #             "/World/target",
        #             "/World/defaultGroundPlane",
        #             "/curobo",
        #         ],
        #         reference_prim_path=robot_prim_path,
        #     )
        #     obstacles.add_obstacle(world_cfg_table.cuboid[0])
        #     mpc.world_coll_checker.load_collision_model(obstacles)

        # # position and orientation of target virtual cube:
        # cube_position, cube_orientation = target.get_world_pose()

        # if past_pose is None:
        #     past_pose = cube_position + 1.0

        # if np.linalg.norm(cube_position - past_pose) > 1e-3:
        #     # Set EE teleop goals, use cube for simple non-vr init:
        #     ee_translation_goal = cube_position
        #     ee_orientation_teleop_goal = cube_orientation
        #     ik_goal = Pose(
        #         position=tensor_args.to_device(ee_translation_goal),
        #         quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
        #     )
        #     goal_buffer.goal_pose.copy_(ik_goal)
        #     mpc.update_goal(goal_buffer)
        #     past_pose = cube_position

        # # if not changed don't call curobo:

        # # get robot current state:
        # sim_js = robot.get_joints_state()
        # js_names = robot.dof_names
        # sim_js_names = robot.dof_names

        # cu_js = JointState(
        #     position=tensor_args.to_device(sim_js.positions),
        #     velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
        #     acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
        #     jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
        #     joint_names=sim_js_names,
        # )
        # cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        # if cmd_state_full is None:
        #     current_state.copy_(cu_js)
        # else:
        #     current_state_partial = cmd_state_full.get_ordered_joint_state(
        #         mpc.rollout_fn.joint_names
        #     )
        #     current_state.copy_(current_state_partial)
        #     current_state.joint_names = current_state_partial.joint_names
        #     # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        # common_js_names = []
        # current_state.copy_(cu_js)

        # mpc_result = mpc.step(current_state, max_attempts=2)
        # # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        # succ = True  # ik_result.success.item()
        # cmd_state_full = mpc_result.js_action
        # common_js_names = []
        # idx_list = []
        # for x in sim_js_names:
        #     if x in cmd_state_full.joint_names:
        #         idx_list.append(robot.get_dof_index(x))
        #         common_js_names.append(x)

        # cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        # cmd_state_full = cmd_state

        # art_action = ArticulationAction(
        #     cmd_state.position.cpu().numpy(),
        #     # cmd_state.velocity.cpu().numpy(),
        #     joint_indices=idx_list,
        # )
        # # positions_goal = articulation_action.joint_positions
        # if step_index % 1000 == 0:
        #     print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        # if succ:
        #     # set desired joint angles obtained from IK:
        #     for _ in range(3):
        #         articulation_controller.apply_action(art_action)

        # else:
        #     carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()