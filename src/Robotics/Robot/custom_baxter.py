from pyrep.robots.end_effectors.gripper import Gripper
import numpy as np

ALLOWED_MOVEMENT = 0.0001
POSITION_ERROR = 0.008

class CustomBaxter(Gripper):
    def __init__(self, count: int=0) -> None:
        super().__init__(count, 'BaxterGripper',
                         ['BaxterGripper_closeJoint'])
        # interval_close = [-0.1, 0.1]
        self.interval_close = [-0.0637, 0.1]
        self.max_torque = 20.0

        cyclics, _ = self.get_joint_intervals()
        self.set_joint_intervals(cyclics, [self.interval_close])

    def holdTight(self, velocity: float) -> bool:
        """Actuate the gripper, but return after each simulation step.

        The functions attempts to open/close the gripper according to 'amount',
        where 1 represents open, and 0 represents close. The user should
        iteratively call this function until it returns True.

        This is a convenience method. If you would like direct control of the
        gripper, use the :py:class:`RobotComponent` methods instead.

        For some grippers, this method will need to be overridden.

        :param amount: A float between 0 and 1 representing the gripper open
            state. 1 means open, whilst 0 means closed.
        :param velocity: The velocity to apply to the gripper joints.


        :raises: ValueError if 'amount' is not between 0 and 1.

        :return: True if the gripper has reached its open/closed limits, or if
            the 'max_force' has been exerted.

        THIS FUNCTION WAS ADAPTED FROM THE PYREP 'ACTUATE' FUNCTION
        """
        target_pos = self.interval_close[1]
        current_positions = self.get_joint_positions()
        j, cur, prev = self.joints[0], current_positions[0], self._prev_positions[0]

        done = True
        # Check if the joint has moved much
        not_moving = (prev is not None and
                        np.fabs(cur - prev) < ALLOWED_MOVEMENT)
        reached_target = np.fabs(target_pos - cur) < POSITION_ERROR
        vel = -velocity if cur - target_pos > 0 else velocity
        if reached_target or not_moving:
            self._prev_positions = [None] * self._num_joints
            self._prev_vels = [None] * self._num_joints
            j.set_joint_force(30.0)
            j.set_joint_target_velocity(0.1) # to maintain grip
        else:
            done = False
            self._prev_vels[0] = vel  # type: ignore
            j.set_joint_target_velocity(vel)
        self._prev_positions = current_positions  # type: ignore
        return done

    def releaseGrip(self, velocity: float):
        target_pos = self.interval_close[0]
        current_positions = self.get_joint_positions()
        j, cur, prev = self.joints[0], current_positions[0], self._prev_positions[0]

        done = True
        # Check if the joint has moved much
        not_moving = (prev is not None and
                        np.fabs(cur - prev) < ALLOWED_MOVEMENT)
        reached_target = np.fabs(target_pos - cur) < POSITION_ERROR
        vel = -velocity if cur - target_pos > 0 else velocity
        if reached_target or not_moving:
            self._prev_positions = [None] * self._num_joints
            self._prev_vels = [None] * self._num_joints
            j.set_joint_force(self.max_torque)
            self.set_joint_target_velocities([0.0] * self._num_joints) # stop once gripper is fully open
        else:
            done = False
            self._prev_vels[0] = vel  # type: ignore
            j.set_joint_target_velocity(vel)
        self._prev_positions = current_positions  # type: ignore
        return done

    def changeAperture(self, amount: float, velocity: float) -> bool:
        """Actuate the gripper, but return after each simulation step.

        The functions attempts to open/close the gripper according to 'amount',
        where 1 represents open, and 0 represents close. The user should
        iteratively call this function until it returns True.

        This is a convenience method. If you would like direct control of the
        gripper, use the :py:class:`RobotComponent` methods instead.

        For some grippers, this method will need to be overridden.

        :param amount: A float between 0 and 1 representing the gripper open
            state. 1 means open, whilst 0 means closed.
        :param velocity: The velocity to apply to the gripper joints.


        :raises: ValueError if 'amount' is not between 0 and 1.

        :return: True if the gripper has reached its open/closed limits, or if
            the 'max_force' has been exerted.

        THIS FUNCTION WAS ADAPTED FROM THE PYREP 'ACTUATE' FUNCTION
        """
        if not (0.0 <= amount <= 1.0):
            raise ValueError("'open_amount' should be between 0 and 1.'")
        _, joint_intervals_list = self.get_joint_intervals()
        joint_intervals = np.array(joint_intervals_list)

        # Decide on if we need to open or close
        joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
        target_pos = joint_intervals[:, 1] - (joint_range * amount)

        current_positions = self.get_joint_positions()
        done = True
        j, target, cur, prev = self.joints[0], target_pos[0], current_positions[0], self._prev_positions[0]
        # Check if the joint has moved much
        not_moving = (prev is not None and
                        np.fabs(cur - prev) < POSITION_ERROR)
        reached_target = np.fabs(target - cur) < POSITION_ERROR
        vel = -velocity if cur - target > 0 else velocity
        oscillating = (self._prev_vels[0] is not None and
                        vel != self._prev_vels[0])
        if reached_target or not_moving: #or oscillating:
            j.set_joint_target_velocity(0)
        else:
            done = False
            self._prev_vels[0] = vel  # type: ignore
            j.set_joint_target_velocity(vel)
        self._prev_positions = current_positions  # type: ignore
        if done:
            self._prev_positions = [None] * self._num_joints
            self._prev_vels = [None] * self._num_joints
            self.set_joint_target_velocities([0.0] * self._num_joints)
        return done












# class CustomBaxter(Gripper):
#     def __init__(self, count: int=0) -> None:
#         super().__init__(count, 'BaxterGripper',
#                          ['BaxterGripper_closeJoint',
#                          'BaxterGripper_centerJoint'])
#         # super().__init__(count, 'BaxterGripper',
#         #                  ['BaxterGripper_closeJoint'])
#         cyclics, _ = self.get_joint_intervals()
#         interval_close = [-0.1, 0.1]
#         # interval_close = [-0.0637, 0.1]
#         interval_center = [-1, 1]
#         self.set_joint_intervals(cyclics, [interval_close, interval_center])

#     def changeAperture(self, amount: float, velocity: float) -> bool:
#         """Actuate the gripper, but return after each simulation step.

#         The functions attempts to open/close the gripper according to 'amount',
#         where 1 represents open, and 0 represents close. The user should
#         iteratively call this function until it returns True.

#         This is a convenience method. If you would like direct control of the
#         gripper, use the :py:class:`RobotComponent` methods instead.

#         For some grippers, this method will need to be overridden.

#         :param amount: A float between 0 and 1 representing the gripper open
#             state. 1 means open, whilst 0 means closed.
#         :param velocity: The velocity to apply to the gripper joints.


#         :raises: ValueError if 'amount' is not between 0 and 1.

#         :return: True if the gripper has reached its open/closed limits, or if
#             the 'max_force' has been exerted.

#         THIS FUNCTION WAS ADAPTED FROM THE PYREP 'ACTUATE' FUNCTION
#         """
#         if not (0.0 <= amount <= 1.0):
#             raise ValueError("'open_amount' should be between 0 and 1.'")
#         _, joint_intervals_list = self.get_joint_intervals()
#         joint_intervals = np.array(joint_intervals_list)

#         # Decide on if we need to open or close
#         joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
#         target_pos = joint_intervals[:, 1] - (joint_range * amount)

#         current_positions = self.get_joint_positions()
#         done = True
#         j, target, cur, prev = self.joints[0], target_pos[0], current_positions[0], self._prev_positions[0]
#         # Check if the joint has moved much
#         not_moving = (prev is not None and
#                         np.fabs(cur - prev) < POSITION_ERROR)
#         reached_target = np.fabs(target - cur) < POSITION_ERROR
#         vel = -velocity if cur - target > 0 else velocity
#         oscillating = (self._prev_vels[0] is not None and
#                         vel != self._prev_vels[0])
#         if not_moving or reached_target or oscillating:
#             j.set_joint_target_velocity(0)
#         else:
#             done = False
#             self._prev_vels[0] = vel  # type: ignore
#             j.set_joint_target_velocity(vel)
#         self._prev_positions = current_positions  # type: ignore
#         if done:
#             self._prev_positions = [None] * self._num_joints
#             self._prev_vels = [None] * self._num_joints
#             self.set_joint_target_velocities([0.0] * self._num_joints)
#         return done