#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


# script running (ubuntu):
#

############################################################


# Third Party
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
parser.add_argument("--robot2", type=str, default="ur10e.yml", help="robot configuration to load")
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

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

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


def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


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
    # target=cuboid.DynamicCuboid(
    #      "/World/target",
    #     # mass=None,
    #     position=np.array([0.5, 0, 0.5]),
    #     orientation=np.array([0, 1, 0, 0]),
    #     color=np.array([1.0, 0, 0]),
    #     size=0.05,
    #     linear_velocity=[0.0, 1, 9.0],

    # )
    target_2= cuboid.VisualCuboid(
        "/World/target1",
        position=np.array([0.75, 0.75, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 1, 1]),
        size=0.05,
    )
    setup_curobo_logger("warn")
    past_pose = None
    past_pose2 = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None
    target_pose2 = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    robot2_cfg= load_yaml(join_path(get_robot_configs_path(), args.robot2))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    j_names2 = robot2_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    default2_config = robot2_cfg["kinematics"]["cspace"]["retract_config"]

    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02
    robot2_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    robot2, robot2_prim_path = add_robot_to_scene(robot2_cfg, my_world,robot_name='robot2',position=[0.5,0.5,0])

    articulation_controller = robot.get_articulation_controller()
    articulation_controller2 = robot2.get_articulation_controller()

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
    # =world_file

    init_curobo = False

    # tensor_args = TensorDeviceType()

    # robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg1 = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # ).get_mesh_world()
    # world_cfg1.mesh[0].pose[2] = -10.0

    # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    # j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    # default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=False,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )
    mpc2_config = MpcSolverConfig.load_from_robot_config(
        robot2_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=False,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )

    mpc = MpcSolver(mpc_config)
    mpc2=MpcSolver(mpc2_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    retract_cfg2 = mpc2.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)

    joint_names = mpc.rollout_fn.joint_names
    joint_names2 = mpc2.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    state2 = mpc2.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg2, joint_names=joint_names2)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    current_state2=JointState.from_position(retract_cfg2, joint_names=joint_names2)

    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    retract_pose2 = Pose(state2.ee_pos_seq, quaternion=state2.ee_quat_seq)  

    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )
    
    goal2 = Goal(
        current_state=current_state2,
        goal_state=JointState.from_position(retract_cfg2, joint_names=joint_names2),
        goal_pose=retract_pose2,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    goal_buffer2 = mpc2.setup_solve_single(goal2, 1)

    mpc.update_goal(goal_buffer)
    mpc2.update_goal(goal_buffer2)

    mpc_result = mpc.step(current_state, max_attempts=1)
    mpc2_result = mpc2.step(current_state2, max_attempts=1)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    cmd_state_full2 = None

    step = 0
    step2 = 0
    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())
        draw_points(mpc2.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 2:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            idx_list2=[robot2.get_dof_index(x) for x in j_names2]

            robot.set_joint_positions(default_config, idx_list)
            robot2.set_joint_positions(default2_config, idx_list2)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            robot2._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list2))]), joint_indices=idx_list2
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step2+=1
        step_index = step
        step2_index=step2
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)
            # mpc2.world_coll_checker.load_collision_model(obstacles)
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles2 = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                ignore_substring=[
                    robot2_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot2_prim_path,
            )
            obstacles2.add_obstacle(world_cfg_table.cuboid[0])
            mpc2.world_coll_checker.load_collision_model(obstacles2)
        

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()
        cube_position2, cube_orientation2 = target_2.get_world_pose()

        if past_pose is None:
            past_pose = cube_position + 1.0
        if past_pose2 is None:
            past_pose2 = cube_position2 + 1.0

        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position
        if np.linalg.norm(cube_position2 - past_pose2) > 1e-3:
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position2
            ee_orientation_teleop_goal = cube_orientation2
            ik_goal2 = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_buffer2.goal_pose.copy_(ik_goal2)
            mpc2.update_goal(goal_buffer2)
            past_pose2 = cube_position2
        # if not changed don't call curobo:

        # get robot current state:
        sim_js = robot.get_joints_state()
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        sim_js2 = robot2.get_joints_state()
        sim_js2_names = robot2.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)

        cu_js2 = JointState(
            position=tensor_args.to_device(sim_js2.positions),
            velocity=tensor_args.to_device(sim_js2.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js2.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js2.velocities) * 0.0,
            joint_names=sim_js2_names,
        )
        cu_js2 = cu_js2.get_ordered_joint_state(mpc2.rollout_fn.joint_names)
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        if cmd_state_full2 is None:
            current_state2.copy_(cu_js2)
        else:
            current_state_partial2 = cmd_state_full2.get_ordered_joint_state(
                mpc2.rollout_fn.joint_names
            )
            current_state2.copy_(current_state_partial2)
            current_state2.joint_names = current_state_partial2.joint_names

        # common_js_names = []

        current_state.copy_(cu_js)
        current_state2.copy_(cu_js2)

        mpc_result = mpc.step(current_state, max_attempts=1)
        mpc2_result=mpc2.step(current_state2, max_attempts=1)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        cmd_state_full2 = mpc2_result.js_action

        common_js_names = []
        common_js2_names=[]

        idx_list = []
        idx2_list=[]
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        for x in sim_js2_names:
            if x in cmd_state_full2.joint_names:
                idx2_list.append(robot2.get_dof_index(x))
                common_js2_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state2 = cmd_state_full2.get_ordered_joint_state(common_js2_names)

        cmd_state_full = cmd_state
        cmd_state_full2 = cmd_state2

        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        art_action2=ArticulationAction(
            cmd_state2.position.cpu().numpy(),
            joint_indices=idx2_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())
            print(mpc2_result.metrics.feasible.item(), mpc2_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(3):
                articulation_controller.apply_action(art_action)
            for _ in range(3):
                articulation_controller2.apply_action(art_action2)

        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()
