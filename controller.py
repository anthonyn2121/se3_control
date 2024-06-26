import numpy as np
from scipy.spatial.transform import Rotation

def vmap(R):
    """
    Compute the vector map (vmap) of a rotation matrix R.
    
    Args:
        R (numpy.ndarray): 3x3 rotation matrix.
    
    Returns:
        numpy.ndarray: 3x1 vector representing the rotation axis and angle.
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # Handle the case when theta is close to 0 or pi
    if np.isclose(theta, 0):
        return np.zeros(3)
    elif np.isclose(theta, np.pi):
        # Use an approximation for small angles
        rx = (R[2, 1] - R[1, 2]) / 2
        ry = (R[0, 2] - R[2, 0]) / 2
        rz = (R[1, 0] - R[0, 1]) / 2
        return theta * np.array([rx, ry, rz])
    else:
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        omega = theta / (2 * np.sin(theta)) * np.array([rx, ry, rz])
        return omega
    
def hat(omega):
    """
    Compute the hat operator of a 3D vector.
    
    Args:
        omega (numpy.ndarray): 3x1 or 1x3 vector
    
    Returns:
        numpy.ndarray: 3x3 skew-symmetric matrix
    """
    omega = omega.flatten()  # Ensure omega is a 1D array
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])


class SE3Control(object):
    def __init__(self, params):
        # Quadrotor physical parameters.
        self.mass            = params['mass'] # kg
        self.Ixx             = params['Ixx']  # kg*m^2
        self.Iyy             = params['Iyy']  # kg*m^2
        self.Izz             = params['Izz']  # kg*m^2
        self.arm_length      = params['arm_length'] # meters
        self.rotor_speed_min = params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = params['rotor_speed_max'] # rad/s
        self.k_thrust        = params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = params['k_drag']   # Nm/(rad/s)**2
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2

        self.g = 9.81 # m/s^2

        self.e3 = np.array([0, 0, 1])
        self.Kx = np.ones(3)
        self.Kv = np.ones(3)
        self.KR = np.ones((3,3))
        self.KOmega = np.ones((3,3))

    def update(self, dt, state, desired_state):
        
        ## Track position and velocity error
        ex = state['x'] - desired_state['x']  ## position error
        ev = state['x_dot'] - desired_state['x_dot']  ## velocity error
        
        ## Find desired headings
        b1d = np.array([1, 0, 0])  ## vector in inertial frame indicating where nose of quadrotor should point -- nose aligns with desired path

        b3d = -(self.Kx*ex) - (self.Kv*ev) - (self.mass*self.g*self.e3) + (self.mass*desired_state['x_ddot'])
        b3d /= (np.linalg.norm(b3d) + 0.0001)
        print("b3d: \n", b3d)

        b2d = np.cross(b3d, b1d)
        b2d /= np.linalg.norm(b2d)
        print("b2d: \n", b2d)

        b1d = np.cross(b2d, b3d)
        print("b1d: \n", b1d)

        Rd = np.vstack((b1d, b2d, b3d))  ## desired attitude as a orthogonal group in SO(3)
        print("Rd: \n", Rd)
        ## Tracking rotation and angular velocity error
        R = Rotation.from_matrix(state['R']).as_matrix()  ## current rotation
        print("R.shape: \n", R.shape)
        eR = 0.5 * vmap(Rd @ R - R @ Rd)
        eOmega = state['omega']  ## desired angular velocity is 0, so the equation (omega - R @ Rd @ Omegad) == 0

        ## Find control inputs and update dynamics
        f = (-self.Kx*ex - self.Kv*ev - self.mass*self.g*self.e3) + self.mass*desired_state['x_ddot']  ## total thrust
        M = -self.KR*eR - self.KOmega*eOmega + np.cross(state['omega'], self.inertia*state['omega'])  ## control moment

        x = state['x'] + state['x_dot'] * dt
        v = state['x_dot'] + ((1/self.mass)*(self.mass*self.g*self.e3 - f*R@self.e3)) * dt
        orientation = R + (R @ hat(state['omega'])) * dt
        angular_velocity = state['omega'] + np.linalg.inv(self.inertia) @ (M - np.cross(state['omega'], self.inertia@state['omega'])) * dt

        state = {'x': x.reshape((1, 3)),
                 'x_dot': v.reshape((1, 3)),
                 'R': orientation.reshape((3, 3)),
                 'omega': angular_velocity.reshape((3, 3))}
        return state