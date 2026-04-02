import sys
import os
from math import *

sys.path.append(os.path.abspath("examples"))

import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate
from traj_gen import CubicPolynomial, QuinticPolynomial, Trapezoidal



class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        curr_joint_values = joint_values.copy()
        
        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]
        
        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

        # Set the Denavit-Hartenberg parameters for each joint
        DH = np.zeros((self.num_dof, 4)) # [theta, d, a, alpha]
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist
    
    def calc_jacobian(self, joint_values: list, radians=True):
        """
        Calculate the Jacobian of linear velocity for the HiWonder arm.

        Args:
            joint_values (list): Current joint angles in radians.
            radians (bool): Whether the input joint angles are in radians.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        curr_joint_values = joint_values.copy()
        
        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]
        
        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
            
        DH = np.zeros((self.num_dof, 4)) # [theta, d, a, alpha]
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]
        #DH = self.calc_dh(joint_values, radians=radians)
        
        H_LIST = [ut.dh_to_matrix(DH[i]) for i in range(len(joint_values))]
        H_01, H_12, H_23, H_34, H_45 = H_LIST
        H_EE = H_01@H_12@H_23@H_34@H_45  # Final transformation matrix for EE

        H_02 = H_01@H_12
        H_03 = H_02@H_23
        H_04 = H_03@H_34

        d_EE = H_EE[0:3, 3]
        k = np.array([0, 0, 1])
        jacobian = np.zeros(shape=(3, len(joint_values)))

        for i, H in enumerate([None, H_01, H_02, H_03, H_04]):
            if H is None:
                Jv = np.cross(k, d_EE)
            else:
                z = H[0:3, 0:3]@k
                r = d_EE - H[0:3, 3]
                Jv = np.cross(z, r)
            jacobian[:, i] = Jv.T

        return jacobian

    def calc_inv_jacobian(self, joint_values: list, lambda_: float = 0.05):
        """
        Calculate the pseudo-inverse of the Jacobian with dampening.

        Using the formula: J^T * (J * J^T + lambda^2 * I)^(-1)
        where lambda is the dampening factor to avoid singularities.

        Args:
            joint_values (list): Current joint angles in radians.
            lambda_ (float): Dampening factor
        
        Returns:
            np.ndarray: The pseudo-inverse of the Jacobian matrix.

        """
        J = self.calc_jacobian(joint_values)
        return J.T@ np.linalg.inv(J@J.T + (lambda_**2)*np.eye(J.shape[0]))

    def calc_ik_single_soln(self, ee: ut.EndEffector, joint_values: ut.List[float], soln: int = 0):
        """
        Calculate the inverse kinematics for the HiWonder arm.
        NOTE: There will be 4 solutions for the arm

        Args:
            ee (EndEffector): Desired end effector position and orientation.
            joint_values (list): Current joint angles in radians.
            soln (int): Solution index for multiple IK solutions (if applicable).
        """
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        th1_config = soln // 2
        elbow_config = soln % 2

        # Step 1: Compute wrist position
        R_05 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
        d5 = l4 + l5
        p_ee = np.array([ee.x, ee.y, ee.z])
        p_wrist = p_ee - d5*R_05@np.array([0,0,1])
        wx,wy,wz = p_wrist

        # Step 2: Compute theta 1-3

        ## Theta 1
        if th1_config == 0:
            th1 = atan2(wy, wx)
            r = sqrt(wx**2 + wy**2)
        else:
            th1 = atan2(-wy, -wx)
            r = -sqrt(wx**2 + wy**2)

        ## Theta 3
        S = wz - l1
        L = sqrt(r**2 + S**2)
        try:
            beta_cos = (l2**2 + l3**2 - L**2)/(2*l2*l3)
            beta = acos(beta_cos)
        except ValueError:
            print("Target is out of reach for the arm.")
            return joint_values

        if elbow_config == 0:
            th3 = pi - beta
        else:
            th3 = beta - pi

        ## Theta 2
        psi = atan2(S, r)
        alpha = atan2(l3*sin(th3), l2+l3*cos(th3))
        th2 = ut.wraptopi(pi/2 - psi + alpha)
        
        # Step 3: Compute R_03
        R_01 = ut.dh_to_matrix([th1, l1, 0, -pi/2])[0:3, 0:3]
        R_12 = ut.dh_to_matrix([th2-pi/2, 0, l2, pi])[0:3, 0:3]
        R_23 = ut.dh_to_matrix([th3, 0, l3, pi])[0:3, 0:3]
        R_03 = R_01 @ R_12 @ R_23

        # Step 4: Compute R_35
        R_35 = R_03.T @ R_05

        # Step 5: Compute theta 4-5

        ## Theta 4
        sin_th4 = R_35[1,2]
        cos_th4 = R_35[0,2]
        th4 = atan2(sin_th4, cos_th4)

        ## Theta 5
        sin_th5 = R_35[2,0]
        cos_th5 = R_35[2,1]
        th5 = atan2(sin_th5, cos_th5)   
        return [th1, th2, th3, th4, th5]

    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values: ut.List[float], soln: int = 0):
        soln_idx = [soln, (soln+1)%4, (soln+2)%4, (soln+3)%4]
        for i in soln_idx:
            soln_candidate = self.calc_ik_single_soln(ee, joint_values, soln = i)
            if ut.check_valid_ik_soln(soln_candidate, ee, self):
                return soln_candidate
        print("No valid IK solution found for the given end effector pose.")

    def calc_numerical_ik(self, ee: ut.EndEffector, joint_values: ut.List[float], tol: float = 0.002, ilimit: int = 1000):

        # Arm is underactuated, only do position IK
        p_ee = np.array([ee.x, ee.y, ee.z])
        # Get initial guess of joint values; ensure they are sampled from valid joint values.
        lim = [
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3],
                [-2*np.pi / 3, 2*np.pi / 3]
              ]
        if joint_values is not None:
            print(f"Initial guess provided {joint_values}")
            guess = np.array(joint_values, dtype=float)
        else:
            guess = np.array([
                    random.uniform(*lim[0]), 
                    random.uniform(*lim[1]),
                    random.uniform(*lim[2]),
                    random.uniform(*lim[3]),
                    random.uniform(*lim[4]),
                    ], dtype=float)
        icount = 0
        while icount < ilimit:
            fk_result, _ = self.calc_forward_kinematics(guess, True)
            diff = p_ee - np.array([fk_result.x, fk_result.y, fk_result.z])
            if np.linalg.norm(diff) < tol and ut.check_joint_limits(guess, lim):
                return guess
            guess += self.calc_inv_jacobian(guess) @ diff
            icount += 1
        return guess



if __name__ == "__main__":
    
    robot_model = FiveDOFRobot()
    traj_model = QuinticPolynomial()
    
    robot = RobotSim(robot_model=robot_model, traj_model=traj_model)
    viz = Visualizer(robot=robot)
    viz.run()