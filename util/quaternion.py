import math
import numpy as np

def inverse_quaternion(quaternion):
    result = np.copy(quaternion)
    result[1:4] = -result[1:4]
    return result

def quaternion_product(q1, q2):
    result = np.zeros(4)
    result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
    result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    if result[0] < 0:
        result = -result
    return result

def quaternion_distance(q1: np.ndarray, q2: np.ndarray):
    """
    Returns a distance measure between two quaternions. Returns 0 whenever the quaternions represent
    the same orientation and gives 1 when the two orientations are 180 degrees apart. Note that this
    is NOT a quaternion difference; the difference q_diff implies q1 * q_diff = q2 and is NOT
    commutative. This uses the fact that if q1 and q2 are equal, the q1 * q2 is equal to 1 (or -1 if
    they only differ in sign) and IS commutative.

    Arguments:
    q1 (numpy ndarray): first quaternion to compare
    q2 (numpy ndarray): second quaternion to compare to
    """
    assert q1.shape == (4,), \
        f"quaternion_similarity received quaternion 1 of shape {q1.shape}, but should be of shape (4,)"
    assert q2.shape == (4,), \
        f"quaternion_similarity received quaternion 2 of shape {q2.shape}, but should be of shape (4,)"
    return 1 - np.inner(q1, q2) ** 2

def rotate_by_quaternion(vector, quaternion):
    q1 = np.copy(quaternion)
    q2 = np.zeros(4)
    q2[1:4] = np.copy(vector)
    q3 = inverse_quaternion(quaternion)
    q = quaternion_product(q2, q3)
    q = quaternion_product(q1, q)
    result = q[1:4]
    return result

def quaternion2euler(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    ysqr = y * y

    ti = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(ti, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    result = np.zeros(3)
    result[0] = X * np.pi / 180
    result[1] = Y * np.pi / 180
    result[2] = Z * np.pi / 180

    return result

def euler2quat(z=0, y=0, x=0):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result = np.array([
        cx*cy*cz - sx*sy*sz,
        cx*sy*sz + cy*cz*sx,
        cx*cz*sy - sx*cy*sz,
        cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
        result = -result
    return result


def euler2so3(z=0, y=0, x=0):

    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)],
    ])

    R_y = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)],
    ])

    R_z = np.array([
        [math.cos(z), -math.sin(z), 0,],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1],
    ])

    return R_z @ R_y @ R_x

def add_euler(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]