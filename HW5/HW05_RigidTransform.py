import numpy as np

# 원 좌표

p1 = np.array([-0.5,0,2.121320])

p2 = np.array([0.5,0,2.121320])

p3 = np.array([0.5,-0.707107,2.828427])

p4 = np.array([0.5, 0.707107, 2.828427])

p5 = np.array([1.0, 1.0, 1.0])

# 변환된 좌표

p1_p = np.array([1.363005,-0.427130,2.339082])

p2_p = np.array([1.748084,0.437983,2.017688])

p3_p = np.array([2.636461,0.184843,2.400710])

p4_p = np.array([1.4981, 0.8710, 2.8837])

# 크기가 같은 임의의 두 벡터 외적을 통해 구하기

h = np.cross(p2-p1,p3-p1)

h_unit = h / (np.linalg.norm(h))

h_p = np.cross(p2_p-p1_p, p3_p-p1_p)

h_p_unit = h_p / (np.linalg.norm(h_p))

# 임의의 두 벡터의 내적과 외적을 통해 회전축과 각도를 구한다.

r1_axis = np.cross(h_unit,h_p_unit)

r1_axis_unit = r1_axis / (np.linalg.norm(r1_axis))

r1_cos = np.dot(h_unit,h_p_unit) / (np.linalg.norm(h_unit)*np.linalg.norm(h_p_unit))

# 구해진 값으로 R1을 구한다.

def rotational_transform(axis, cos):

    sin = np.sqrt(1 - (cos ** 2))

    result = np.array([[cos + (axis[0]**2)*(1-cos), axis[0]*axis[1]*(1-cos) - axis[2]*sin, axis[0]*axis[2]*(1-cos) + axis[1]*sin],
                       [axis[1]*axis[0]*(1-cos) + axis[2]*sin, cos + (axis[1]**2)*(1-cos), axis[1]*axis[2]*(1-cos) - axis[0]*sin],
                       [axis[2]*axis[0]*(1-cos) - axis[1]*sin, axis[2]*axis[1]*(1-cos) + axis[0]*sin, cos + (axis[2]**2)*(1-cos)]])
    return result

r1 = rotational_transform(r1_axis_unit,r1_cos)

# R2를 계산한다.

r2_axis = h_p_unit

r2_cos = np.dot((np.matmul(r1,p3-p1)), (p3_p-p1_p)) / (np.linalg.norm(np.matmul(r1,p3-p1)) * np.linalg.norm(p3_p-p1_p))

r2 = rotational_transform(r2_axis,r2_cos)

# rigid 변환을 한다.
def rigid_transform(axis):

    result = np.matmul(np.matmul(r1, np.transpose(axis - p1)), r2) + p1_p

    return result

# 검증과 테스트를 한다.

print("p4 변환된 예측값 : ", rigid_transform(p4), "p4 정답 좌표값 : " , p4_p)

print("본 p5 좌표값 : ", p5, "변환된 p5 좌표값 : ", rigid_transform(p5))

