# -*- coding: utf-8 -*-


import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import time


class BoxEnv(gym.Env):
    """
    功能：如下环境封装为 gym env，用于验证强化学习 action masking。

    目标：将地图中的一个箱子移动到目的地

    环境配置参数：二维数组表示地图，1为箱子，0为空地，-1为目的地。示例

    [[0, 0, 1, 0],

    [0, 0, 0, 0],

    [0, -1, 0, 0]]

    note: 1的位置为（2，0），numpy[0][2]

    观测空间： 当前地图数组

    动作空间：三元组 (y, x, d)，将位于 (x、y) 的箱子往 d 方向移动一格（0123对应上右下左）。如果 (x,y) 处无箱子则不行动。

    奖励函数：每走一步reward为 -1。

    """

    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 2
    # }

    def __init__(self, map):
        # mat=mat_dict["mat"]

        self.target_y = 0
        self.target_x = 0
        self.start_mat = np.array(map)
        self.map = np.array(map)
        self.h, self.w = self.map.shape
        self.viewer = None
        self.reward = 0
        self.reward_sum = 0
        self.step_count = 0
        self.step_real_count = 0
        for i in range(self.h):
            for j in range(self.w):
                if map[i][j] == -1:
                    self.target_y = i
                    self.target_x = j

        self.observation_space = spaces.Box(-1, 1, shape=(self.h, self.w), dtype=np.int)

        # [h,w,4]->(y,x,action)
        self.action_space = spaces.MultiDiscrete([self.h, self.w, 4])

    def step(self, action):

        if self.map[action[0]][action[1]] == 1:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            self.step_count += 1

            # 适合多个箱子
            state = np.array([action[0], action[1]])
            action = action[2]

            offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            # print("方向{}".format(offsets[action]))
            y, x = state + offsets[action]

            # h对应x，w对应y
            if self.h > y >= 0 and self.w > x >= 0:
                # 移动前的位置置0
                self.map[state[0]][state[1]] = 0
                # 当前箱子位置置1
                self.map[y][x] = 1

                self.step_real_count += 1
                state = np.array([y, x])
                # self.counts += 1

            now_dist = abs(state[0] - self.target_y) + abs(state[1] - self.target_x)
            done = (now_dist == 0)
            self.reward = -1

            return self.map, self.reward, done, {}
        reward = -1
        done = False
        return self.map, reward, done, {}

    def reset(self):

        self.reward = 0
        self.reward_sum = 0
        self.step_count = 0
        self.step_real_count = 0
        self.map = np.copy(self.start_mat)
        return self.map

    def render(self, mode='human'):

        WIDTH = 800
        HEIGHT = 800

        out = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
        draw = ImageDraw.Draw(out)
        self.draw_state(draw, self.map, WIDTH, HEIGHT, 0)
        # draw_state()

        rgb_array = np.array(out)
        if mode == 'rgb_array':
            return rgb_array
        else:
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                # from rendering import SimpleImageViewer
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(rgb_array)

    def close(self):
        return None

    def draw_state(self, draw, map, dx, dy, color_schema):

        # 定义颜色，soft 为 1 ，softer 为 2
        GRAY_1 = ImageColor.getrgb("#000000")  # soft gray
        BLUE_1 = ImageColor.getrgb("#56B4E9")  # soft blue
        BLUE_2 = ImageColor.getrgb("#9CCAE4")  # softer blue
        PLAYER1UNIT_OUTLINE = ImageColor.getrgb("#BF3682")  # soft red
        RED_2 = ImageColor.getrgb("#C183A6")  # softer red
        PURPLE_2 = ImageColor.getrgb("#E69F00")  # softer purple
        GREEN_1 = ImageColor.getrgb("#009E73")  # soft green
        ORANGE_1 = ImageColor.getrgb("#D55E00")  # soft orange
        CYAN_1 = ImageColor.getrgb("#0072B2")  # soft cyan
        YELLOW_1 = ImageColor.getrgb("#F0E442")  # soft yellow
        GRAY = (128, 128, 128)
        GREEN = (0, 255, 0)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        h, w = map.shape
        for i in range(h):
            for j in range(w):
                if map[i][j] == 1:
                    now_y = i
                    now_x = j

        # 防止box移动到目的地将target覆盖
        target_y = now_y
        target_x = now_x
        for i in range(h):
            for j in range(w):
                if map[i][j] == -1:
                    target_y = i
                    target_x = j
        gridx = (dx - 64) / w
        gridy = (dy - 64) / h
        grid = min(gridx, gridy)
        info = f"now  x: {now_x} , y: {now_y}"
        draw.text((10, dy - 45), info, fill=CYAN_1)

        info = f"step_sum: {self.step_count} , step_real: {self.step_real_count}"
        draw.text((10, dy - 30), info, fill=PURPLE_2)

        self.reward_sum += self.reward
        info = f"reward_now: {self.reward},reward_sum: {self.reward_sum}"
        draw.text((10, dy - 15), info, fill=GREEN_1)

        gridline_color = BLACK
        for i in range(w + 1):
            draw.line([(i * grid, 0), (i * grid, h * grid)], fill=gridline_color)

        for i in range(h + 1):
            draw.line([(0, i * grid), (w * grid, i * grid)], fill=gridline_color)

        playerColor = PLAYER1UNIT_OUTLINE
        ucolor = ORANGE_1
        reduction = grid / 8
        x0 = target_x * grid + reduction
        y0 = target_y * grid + reduction
        x1 = x0 + grid - reduction * 2
        y1 = y0 + grid - reduction * 2
        draw.ellipse([x0, y0, x1, y1],
                     fill=ucolor,
                     outline=playerColor,
                     width=4)

        ucolor = GREEN

        reduction = grid / 8
        x0 = now_x * grid + reduction
        y0 = now_y * grid + reduction
        x1 = x0 + grid - reduction * 2
        y1 = y0 + grid - reduction * 2
        draw.ellipse([x0, y0, x1, y1],
                     fill=ucolor,
                     outline=playerColor,
                     width=4)

        text_x = now_x * grid + grid / 2
        text_y = now_y * grid + grid / 2

        text = f"now"
        draw.text((text_x, text_y), text, fill=BLACK, anchor="ms")

        text_x = target_x * grid + grid / 2
        text_y = target_y * grid + grid / 2
        text = f"target"
        draw.text((text_x, text_y), text, fill=YELLOW_1, anchor="ms")


if __name__ == '__main__':
    import numpy as np

    env = BoxEnv([[1, 0, 0], [0, 0, -1]])
    env.reset()
    env.render()
    time.sleep(2)
    # 向右移动
    print(env.step([0, 0, 1]))
    env.render()
    time.sleep(2)
    # 向右移动
    print(env.step([0, 1, 1]))
    # 向右移动 移出边界的情况
    env.render()
    time.sleep(2)
    print(env.step([0, 2, 1]))
    # 向下移动，到达目的地
    env.render()
    time.sleep(2)
    print(env.step([0, 2, 2]))
    env.render()
    time.sleep(3)
    print(env.reset())

    env.render()
    time.sleep(3)
