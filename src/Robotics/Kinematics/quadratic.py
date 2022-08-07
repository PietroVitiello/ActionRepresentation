from pyrep.objects import Object, Shape, Dummy
from pyrep.backend import sim
import numpy as np
import math
from typing import Union

from ..target import Target

class Quadratic():

    def __init__(self, tip: Dummy, target: Shape, max_deviation: float=0.05, always_maxDev: bool=True) -> None:
        self.ik_tip = tip
        self.ik_target = target
        self.tip = tip.get_position()
        self.target = target.get_position()
        self.max_deviation = max_deviation
        self.always_maxDev = always_maxDev # whether to always generate trajectories with max deviation

        self.distance_vec = self.target - self.tip
        self.distance = np.linalg.norm(self.distance_vec)

        linear_mid_pos = (self.target + self.tip)/2
        self.linear_mid = Dummy.create(0.001)
        self.set_linearMid(linear_mid_pos)

        self.ortho, self.apex, self.eq = None, None, None
        self.all_dummies = [self.linear_mid]

        self.target_pos = self.tip
        n = Dummy.create(0.05)
        n.set_position(self.tip)
        self.verga = n
        # self.all_dummies.append(self.verga)

    def setTarget(self, target: Union[Target, Dummy]):
        self.ik_target = target
        self.target = target.get_position()
        self.remove_dummies()
        print("Changed quadratic target and removed all previous dummies")
        # self.resetCurve()

    def set_linearMid(self, pos: np.ndarray) -> None:
        self.linear_mid.set_position(pos)
        direction = self.distance_vec / self.distance
        gamma = np.arctan2(direction[1], direction[0])
        x_projection = direction[0]/np.cos(gamma)
        beta = np.arctan2(-direction[2], x_projection)

        self.linear_mid.rotate([0, 0, gamma])
        self.linear_mid.rotate([0, beta, 0])

    def resetCurve(self):
        self.tip = self.ik_tip.get_position()
        self.target = self.ik_target.get_position()

        self.target_pos = self.tip
        self.verga.set_position(self.tip)

        self.distance_vec = self.target - self.tip
        self.distance = np.linalg.norm(self.distance_vec)

        linear_mid_pos = (self.target + self.tip)/2
        self.linear_mid = Dummy.create(0.001)
        # sim.simSetObjectInt32Parameter(self.linear_mid.get_handle(), 10, 11)
        self.set_linearMid(linear_mid_pos)

        self.ortho, self.apex, self.eq = None, None, None
        self.all_dummies.append(self.linear_mid)

    def get_FaceTargetOrientation(self, look_at=None):
        # print(type(None))
        # print(type(look_at) == type(None))
        if type(look_at) == type(None):
            look_at = self.ik_target

        dummy = Dummy.create(0.001)
        direction = look_at.get_position() - self.ik_tip.get_position()
        direction = direction / np.linalg.norm(direction)
        gamma = np.arctan2(direction[1], direction[0])
        x_projection = direction[0]/np.cos(gamma)
        beta = np.arctan2(-direction[2], x_projection)

        dummy.rotate([0, 0, gamma])
        dummy.rotate([0, beta, 0])
        orientation = dummy.get_orientation(relative_to=self.ik_tip)
        # orientation = dummy.get_orientation()
        dummy.remove()
        return orientation

    def get_axis(self, ax=2):
        R = self.linear_mid.get_matrix()
        return R[:-1,ax].squeeze()

    def find_quadratic(self):
        intersection = self.distance/2
        a = - self.apex/(intersection**2)
        c = self.apex
        return np.array([a, c])

    def get_arcLen(self):
        a = self.distance/2
        param = self.eq[0]**2
        sqrt = np.sqrt(4*param*(a**2) + 1)
        hyper_term = np.arcsinh(2*self.eq[0]*a) / self.eq[0]
        return (2*a*sqrt + hyper_term)/2
        # return self.distance

    def getGradient(self):
        rel_pos = self.ik_tip.get_position(relative_to=self.linear_mid)
        # print("relative taget position: ", rel_pos)
        # print(f"\nrelative position: {rel_pos}")
        # print(f"distance/2: {np.linalg.norm(self.distance_vec)/2}")
        # T = self.linear_mid.get_matrix()
        # rel_pos = np.matmul(np.linalg.inv(T), np.vstack((pos, 1)))[:-1]
        return self.eq[0] * 2 * rel_pos[0]

    def getGradientAtTarget(self):
        T = self.linear_mid.get_matrix()
        rel_pos = np.linalg.inv(T[:3,:3]) @ (self.target_pos - T[:3,3])
        # print("relative taget position: ", rel_pos)
        return self.eq[0] * 2 * rel_pos[0]

    def find_middlePoint(self):
        theta = np.random.uniform(-(4/6)*np.pi, (4/6)*np.pi)
        # theta = - np.pi
        self.linear_mid.rotate([theta, 0, 0])

        if self.always_maxDev:
            self.apex = self.max_deviation
        else:
            self.apex = np.random.uniform(0, self.max_deviation)
        self.ortho = self.get_axis()
        self.eq = self.find_quadratic()

        # n = Dummy.create(0.07)
        # n.set_orientation(self.linear_mid.get_orientation())
        # n.set_position(self.linear_mid.get_position()+self.apex*self.ortho)
        # self.n = n
        # self.all_dummies.append(self.n)
        # self.apex = self.linear_mid.get_position() + (ortho * deviation)
        return theta

    # def get_tangentVelocity(self, ik_tip: Object, v: np.ndarray):
    #     grad = self.getGradient(ik_tip)
    #     theta = np.arctan(grad)/2
    #     y_axis = self.get_axis(1)
    #     q = np.hstack((np.cos(theta), y_axis*np.sin(theta)))
    #     q_inv = q
    #     q_inv[1:] = -q[1:]
    #     v = np.hstack((0, v))
    #     v = self.quaternionProd(q_inv, self.quaternionProd(v, q))
    #     return v[1:]

    # def get_tangentVelocity(self, v: np.ndarray):
    #     grad = self.getGradient()
    #     # grad = self.getGradientAtTarget()
    #     theta = - np.arctan(grad)
    #     R = self.linear_mid.get_matrix()[:-1,:-1]
    #     rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
    #                          [0, 1, 0],
    #                          [-np.sin(theta), 0, np.cos(theta)]])
    #     self.target_pos = self.target_pos + ((R @ rotation @ np.linalg.inv(R) @ v)*0.05)
    #     self.verga.set_position(self.target_pos)
    #     return (R @ rotation @ np.linalg.inv(R) @ v)

    def get_tangentVelocity(self, v: np.ndarray):
        grad = self.getGradient()
        theta = - np.arctan(grad)
        R = self.linear_mid.get_matrix()[:-1,:-1]
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        return (R @ rotation @ np.linalg.inv(R) @ v)

    def get_enhancedTangentVelocity(self, v: np.ndarray, time: float):
        grad = self.getGradient()
        theta = - np.arctan(grad)
        R = self.linear_mid.get_matrix()[:-1,:-1]
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        rel_v = rotation @ np.linalg.inv(R) @ v
        # rel_v[2] += 0.4 * np.abs(rel_v[2])
        rel_v = R @ rel_v
        # rel_v[2] += 0.4 * np.abs(rel_v[2])
        rel_v[2] += 0.4 * np.abs(rel_v[2])
        return rel_v

    def getVelocity2Target(self, v: np.ndarray):
        #just follow position
        grad = self.getGradientAtTarget()
        theta = - np.arctan(grad)
        R = self.linear_mid.get_matrix()[:-1,:-1]
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        rel_v = R @ rotation @ (np.array([1,0,0])*np.linalg.norm(v))
        self.target_pos = self.target_pos + (rel_v*0.05)
        # self.verga.set_position(self.target_pos)
        return (self.target_pos - self.ik_tip.get_position()) / 0.05

    # def getVelocity2Target(self, v: np.ndarray):
    #     grad = self.getGradientAtTarget()
    #     gg = self.getGradient()
    #     theta = - np.arctan(gg)
    #     print(f"gradients: {- np.arctan(grad)}\t{- np.arctan(gg)}")
    #     R = self.linear_mid.get_matrix()[:-1,:-1]
    #     rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
    #                          [0, 1, 0],
    #                          [-np.sin(theta), 0, np.cos(theta)]])
    #     rel_v = R @ rotation @ np.linalg.inv(R) @ v #(np.array([1,0,0])*np.linalg.norm(v))
    #     print(f"nana: {self.target_pos}\t{self.ik_tip.get_position()}\t{rel_v}")
    #     print(f"peppina: {self.target_pos - self.ik_tip.get_position()}")
    #     self.target_pos = self.target_pos + (rel_v*0.05)
    #     self.verga.set_position(self.target_pos)
    #     print("distance: ", (self.target_pos - self.ik_tip.get_position()), "\n")
    #     return (self.target_pos - self.ik_tip.get_position()) / 0.05, self.verga

    # def getVelocity2Target(self, v: np.ndarray, step):
    #     if step == 0:
    #         self.target_pos = self.ik_tip.get_position()
    #     grad = self.getGradientAtTarget()
    #     gg = self.getGradient()
    #     theta = - np.arctan(grad)
    #     print(f"gradients: {- np.arctan(grad)}\t{- np.arctan(gg)}")
    #     R = self.linear_mid.get_matrix()[:-1,:-1]
    #     rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
    #                          [0, 1, 0],
    #                          [-np.sin(theta), 0, np.cos(theta)]])
    #     rel_v = R @ rotation @ np.linalg.inv(R) @ v #(np.array([1,0,0])*np.linalg.norm(v))
    #     print(f"nana: {self.target_pos}\t{self.ik_tip.get_position()}\t{rel_v}")
    #     print(f"peppina: {self.target_pos - self.ik_tip.get_position()}")
    #     self.target_pos = self.target_pos + (rel_v*0.05)
    #     self.verga.set_position(self.target_pos)
    #     print("distance: ", (self.target_pos - self.ik_tip.get_position()), "\n")
    #     return (self.target_pos - self.ik_tip.get_position()) / 1, self.verga

    # def getVelocity2Target(self, v: np.ndarray, step):
    #     if step == 0:
    #         self.target_pos = self.ik_tip.get_position()
    #     grad = self.getGradientAtTarget()
    #     gg = self.getGradient()
    #     theta = - np.arctan(grad)
    #     print(f"gradients: {- np.arctan(grad)}\t{- np.arctan(gg)}")
    #     R = self.linear_mid.get_matrix()[:-1,:-1]
    #     rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
    #                          [0, 1, 0],
    #                          [-np.sin(theta), 0, np.cos(theta)]])
    #     rel_v = R @ rotation @ np.linalg.inv(R) @ v #(np.array([1,0,0])*np.linalg.norm(v))
    #     print(f"nana: {self.target_pos}\t{self.ik_tip.get_position()}\t{rel_v}")
    #     print(f"peppina: {self.target_pos - self.ik_tip.get_position()}")
    #     self.target_pos = self.target_pos + (rel_v*0.05)
    #     self.verga.set_position(self.target_pos)
    #     print("distance: ", (self.target_pos - self.ik_tip.get_position()), "\n")
    #     return self.target_pos

    # def getVelocity2Target(self, v: np.ndarray, step):
    #     self.target_pos = self.ik_tip.get_position()
    #     grad = self.getGradientAtTarget()
    #     gg = self.getGradient()
    #     theta = - np.arctan(gg)
    #     print(f"gradients: {- np.arctan(grad)}\t{- np.arctan(gg)}")
    #     R = self.linear_mid.get_matrix()[:-1,:-1]
    #     rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
    #                          [0, 1, 0],
    #                          [-np.sin(theta), 0, np.cos(theta)]])
    #     rel_v = R @ rotation @ np.linalg.inv(R) @ v #(np.array([1,0,0])*np.linalg.norm(v))
    #     print(f"nana: {self.target_pos}\t{self.ik_tip.get_position()}\t{rel_v}")
    #     print(f"peppina: {self.target_pos - self.ik_tip.get_position()}")
    #     self.target_pos = self.target_pos + (rel_v*0.05)
    #     self.verga.set_position(self.target_pos)
    #     print("distance: ", (self.target_pos - self.ik_tip.get_position()), "\n")
    #     return (self.target_pos - self.ik_tip.get_position()) / 0.05, self.verga

    @staticmethod
    def quaternionProd(q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        res = np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
        return res

    def remove_dummies(self):
        for dummy in self.all_dummies:
            dummy.remove()
        self.all_dummies = []






