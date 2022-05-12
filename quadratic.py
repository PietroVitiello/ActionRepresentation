from pyrep.objects import Object, Shape, Dummy
from pyrep.backend import sim
import numpy as np
import math

class Quadratic():

    def __init__(self, tip: Dummy, target: Shape, max_deviation=0.1) -> None:
        self.ik_tip = tip
        self.ik_target = target
        self.tip = tip.get_position()
        self.target = target.get_position()
        self.max_deviation = max_deviation

        self.distance_vec = self.target - self.tip
        self.distance = np.linalg.norm(self.distance_vec) * 3

        linear_mid_pos = (self.target + self.tip)/2
        self.linear_mid = Dummy.create(0.001)
        self.set_linearMid(linear_mid_pos)

        self.ortho, self.apex, self.eq = None, None, None
        self.all_dummies = [self.linear_mid]

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

        self.distance_vec = self.target - self.tip
        self.distance = np.linalg.norm(self.distance_vec)

        linear_mid_pos = (self.target + self.tip)/2
        self.linear_mid = Dummy.create(0.001)
        sim.simSetObjectInt32Parameter(self.linear_mid.get_handle(), 10, 11)
        self.set_linearMid(linear_mid_pos)

        self.ortho, self.apex, self.eq = None, None, None
        self.all_dummies.append(self.linear_mid)

    def get_FaceTargetOrientation(self):
        dummy = Dummy.create(0.001)
        direction = self.ik_target.get_position() - self.ik_tip.get_position()
        direction = direction / np.linalg.norm(direction)
        gamma = np.arctan2(direction[1], direction[0])
        x_projection = direction[0]/np.cos(gamma)
        beta = np.arctan2(-direction[2], x_projection)

        dummy.rotate([0, 0, gamma])
        dummy.rotate([0, beta, 0])
        orientation = dummy.get_orientation(relative_to=self.ik_tip)
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
        # print(f"\nrelative position: {rel_pos}")
        # print(f"distance/2: {np.linalg.norm(self.distance_vec)/2}")
        # T = self.linear_mid.get_matrix()
        # rel_pos = np.matmul(np.linalg.inv(T), np.vstack((pos, 1)))[:-1]
        return self.eq[0] * 2 * rel_pos[0]

    def find_middlePoint(self):
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        # theta = 0
        self.linear_mid.rotate([theta, 0, 0])

        self.apex = np.random.uniform(0, self.max_deviation)
        # self.apex = 0.5
        print(f"apex: {self.apex}")
        self.ortho = self.get_axis()
        self.eq = self.find_quadratic()

        # n = Dummy.create(0.07)
        # n.set_orientation(self.linear_mid.get_orientation())
        # n.set_position(self.linear_mid.get_position()+self.apex*self.ortho)
        # self.n = n
        # self.all_dummies.append(self.n)
        # self.apex = self.linear_mid.get_position() + (ortho * deviation)

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

    def get_tangentVelocity(self, v: np.ndarray):
        grad = self.getGradient()
        theta = - np.arctan(grad)
        R = self.linear_mid.get_matrix()[:-1,:-1]
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0],
                             [-np.sin(theta), 0, np.cos(theta)]])
        return (R @ rotation @ np.linalg.inv(R) @ v)

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






