# -*- coding: utf-8 -*-


import gym
from gym import spaces
import numpy as np


class Box2DEnv(gym.Env):
    """
    功能：如下环境封装为 gym env，用于验证强化学习 action masking。

    目标：将地图中的一个箱子移动到目的地

    环境配置参数：二维数组表示地图，1为箱子，0为空地，-1为目的地。示例

    [[0, 0, 1, 0],

    [0, 0, 0, 0],

    [0, -1, 0, 0]]

    观测空间： 当前地图数组

    动作空间：三元组 (col, row, d)，将位于 col、row 的箱子往 d 方向移动一格（0123对应上右下左）。如果 col、row 处无箱子则不行动。

    奖励函数：每走一步reward为 -1。

    """
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 2
    # }

    def __init__(self, mat):

        self.target_col = 0
        self.target_row = 0
        self.start_mat = np.array(mat)
        self.mat = np.array(mat)
        self.h, self.w = self.mat.shape
        for i in range(self.h):
            for j in range(self.w):
                if mat[i][j] == -1:
                    self.target_col = i
                    self.target_row = j

        self.observation_space = spaces.Box(-1,1,shape=(self.h, self.w), dtype=np.int)

        # [h,w,4]->(col,row,action)
        self.action_space = spaces.MultiDiscrete([self.h,self.w,4])

    def step(self, action):

        if self.mat[action[0]][action[1]] == 1:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            # 适合多个箱子
            state = np.array([action[0], action[1]])
            action = action[2]

            offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            # print("方向{}".format(offsets[action]))
            col, row = state + offsets[action]

            # h对应x，w对应y
            if self.h > col >= 0 and self.w > row >= 0:
                # 移动前的位置置0
                self.mat[state[0]][state[1]] = 0
                # 当前箱子位置置1
                self.mat[col][row] = 1

                state = np.array([col, row])
                # self.counts += 1

            now_dist = abs(state[0] - self.target_col) + abs(state[1] - self.target_row)
            done = (now_dist == 0)

            reward = -1

            return self.mat, reward, done, {}
        reward = -1
        done = False
        return self.mat, reward, done, {}


    def reset(self):
        # self.state = np.array([self.start_col,self.start_row])
        self.counts = 0
        self.mat = np.copy(self.start_mat)
        return self.mat

    def render(self, mode='human'):
        return None

    def close(self):
        return None


if __name__ == '__main__':
    env = Box2DEnv([[1,0,0],[0,0,-1]])
    env.reset()
    a=[0,0,1]
    b=[0,1,1]
    c=[0,2,2]

    print(a)
    print(env.step(a))

    print(env.step(b))
    print(env.step(b))
    print(env.step(c))
    print(env.reset())
    print(env.step(a))

    # b=env.action_space.sample()
    # print(b)
    # env.step(b)
