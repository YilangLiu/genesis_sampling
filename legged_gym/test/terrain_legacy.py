# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# import numpy as np
# from numpy.random import choice
# from scipy import interpolate
# import random
# from . import terrain_utils
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# class Terrain:
#     def __init__(self, cfg: LeggedRobotCfg.terrain) -> None:

#         self.cfg = cfg
#         self.type = cfg.mesh_type
#         if self.type in ["none", 'plane']:
#             return
#         self.env_length = cfg.terrain_length
#         self.env_width = cfg.terrain_width
#         self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

#         self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
#         self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
#         self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
#         self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
#         self.num_goals = cfg.num_goals

#         self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
#         self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

#         self.border = int(cfg.border_size/self.cfg.horizontal_scale)
#         self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
#         self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

#         self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
#         if cfg.curriculum:
#             self.curiculum()
#         elif cfg.selected:
#             self.selected_terrain()
#         else:    
#             self.randomized_terrain()   
        
#         self.heightsamples = self.height_field_raw
#         # if self.type=="trimesh":
#         #     self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
#         #                                                                                     self.cfg.horizontal_scale,
#         #                                                                                     self.cfg.vertical_scale,
#         #                                                                                     self.cfg.slope_treshold)
    
#     def randomized_terrain(self):
#         for k in range(self.cfg.num_sub_terrains):
#             # Env coordinates in the world
#             (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

#             choice = np.random.uniform(0, 1)
#             difficulty = np.random.choice([0.5, 0.75, 0.9])
#             terrain = self.make_terrain(choice, difficulty)
#             self.add_terrain_to_map(terrain, i, j)
        
#     def curiculum(self):
#         for j in range(self.cfg.num_cols):
#             for i in range(self.cfg.num_rows):
#                 difficulty = i / self.cfg.num_rows
#                 choice = j / self.cfg.num_cols + 0.001

#                 terrain = self.make_terrain(choice, difficulty)
#                 self.add_terrain_to_map(terrain, i, j)

#     def selected_terrain(self):
#         for j in range(self.cfg.num_cols):
#             for i in range(self.cfg.num_rows):
#                 terrain = terrain_utils.SubTerrain("terrain",
#                                 width=self.width_per_env_pixels,
#                                 length=self.width_per_env_pixels,
#                                 vertical_scale=self.cfg.vertical_scale,
#                                 horizontal_scale=self.cfg.horizontal_scale)
                
#                 # terrain = terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=0.1, platform_size=3.)
#                 # terrain_utils.pyramid_sloped_terrain(terrain, slope=0.4, platform_size=3.)
#                 # terrain = terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
#                 terrain_utils.discrete_obstacles_terrain(terrain, 0.1, 1.0, 2.0, 20, platform_size=3.)
#                 self.add_terrain_to_map(terrain, i, j)
    
#     def make_terrain(self, choice, difficulty):
#         terrain = terrain_utils.SubTerrain(   "terrain",
#                                 width=self.width_per_env_pixels,
#                                 length=self.width_per_env_pixels,
#                                 vertical_scale=self.cfg.vertical_scale,
#                                 horizontal_scale=self.cfg.horizontal_scale)
#         import pdb; pdb.set_trace()
#         slope = difficulty * 0.4
#         step_height = 0.05 + 0.18 * difficulty
#         discrete_obstacles_height = 0.05 + difficulty * 0.2
#         stepping_stones_size = 1.5 * (1.05 - difficulty)
#         stone_distance = 0.05 if difficulty==0 else 0.1
#         gap_size = 1. * difficulty
#         pit_depth = 1. * difficulty
#         if choice < self.proportions[0]:
#             idx = 0
#             if choice < self.proportions[0]/ 2:
#                 idx = 1
#                 slope *= -1
#             terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
#         elif choice < self.proportions[1]:
#             idx = 2
#             terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
#             terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
#         elif choice < self.proportions[3]:
#             idx = 3
#             if choice<self.proportions[2]:
#                 idx = 4
#                 step_height *= -1
#             terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=step_height, platform_size=3.)
#         elif choice < self.proportions[4]:
#             idx = 5
#             num_rectangles = 20
#             rectangle_min_size = 1.
#             rectangle_max_size = 2.
#             terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
#         elif choice < self.proportions[5]:
#             idx = 6
#             parkour_hurdle_terrain(terrain,
#                                    num_stones=self.num_goals - 2,
#                                    stone_len=0.1+0.3*difficulty,
#                                    hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
#                                    pad_height=0,
#                                    x_range=[1.2, 2.2],
#                                    y_range=self.cfg.y_range,
#                                    half_valid_width=[0.4, 0.8],
#                                    )
#             self.add_roughness(terrain)
#         elif choice < self.proportions[6]:
#             idx = 7
#             parkour_step_terrain(terrain,
#                                    num_stones=self.num_goals - 2,
#                                    step_height=step_height,
#                                    x_range=[1.0, 1.8],
#                                    y_range=self.cfg.y_range,
#                                    half_valid_width=[0.5, 1],
#                                    pad_height=0,
#                                    )
#             # self.add_roughness(terrain)
#         else:
#             idx = 8
#             pit_terrain(terrain, depth=pit_depth, platform_size=4.)
#         terrain.idx = idx
#         import pdb; pdb.set_trace()
#         return terrain

#     def add_roughness(self, terrain, difficulty=1):
#         max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
#         height = random.uniform(self.cfg.height[0], max_height)
#         terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)


#     def add_terrain_to_map(self, terrain, row, col):
#         i = row
#         j = col
#         # map coordinate system
#         start_x = self.border + i * self.length_per_env_pixels
#         end_x = self.border + (i + 1) * self.length_per_env_pixels
#         start_y = self.border + j * self.width_per_env_pixels
#         end_y = self.border + (j + 1) * self.width_per_env_pixels
#         self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

#         # env_origin_x = (i + 0.5) * self.env_length
#         env_origin_x = i * self.env_length + 1.0
#         env_origin_y = (j + 0.5) * self.env_width
#         x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
#         x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
#         y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
#         y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)

#         env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
#         self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
#         self.terrain_type[i, j] = terrain.idx
#         self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]

# def gap_terrain(terrain, gap_size, platform_size=1.):
#     gap_size = int(gap_size / terrain.horizontal_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     center_x = terrain.length // 2
#     center_y = terrain.width // 2
#     x1 = (terrain.length - platform_size) // 2
#     x2 = x1 + gap_size
#     y1 = (terrain.width - platform_size) // 2
#     y2 = y1 + gap_size
   
#     terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
#     terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

# def pit_terrain(terrain, depth, platform_size=1.):
#     depth = int(depth / terrain.vertical_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale / 2)
#     x1 = terrain.length // 2 - platform_size
#     x2 = terrain.length // 2 + platform_size
#     y1 = terrain.width // 2 - platform_size
#     y2 = terrain.width // 2 + platform_size
#     terrain.height_field_raw[x1:x2, y1:y2] = -depth

# def parkour_hurdle_terrain(terrain,
#                            platform_len=2.5, 
#                            platform_height=0., 
#                            num_stones=8,
#                            stone_len=0.3,
#                            x_range=[1.5, 2.4],
#                            y_range=[-0.4, 0.4],
#                            half_valid_width=[0.4, 0.8],
#                            hurdle_height_range=[0.2, 0.3],
#                            pad_width=0.1,
#                            pad_height=0.5,
#                            flat=False):
#     goals = np.zeros((num_stones+2, 2))
#     # terrain.height_field_raw[:] = -200
    
#     mid_y = terrain.length // 2  # length is actually y width

#     dis_x_min = round(x_range[0] / terrain.horizontal_scale)
#     dis_x_max = round(x_range[1] / terrain.horizontal_scale)
#     dis_y_min = round(y_range[0] / terrain.horizontal_scale)
#     dis_y_max = round(y_range[1] / terrain.horizontal_scale)

#     # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
#     half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
#     hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
#     hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

#     platform_len = round(platform_len / terrain.horizontal_scale)
#     platform_height = round(platform_height / terrain.vertical_scale)
#     terrain.height_field_raw[0:platform_len, :] = platform_height

#     stone_len = round(stone_len / terrain.horizontal_scale)
#     # stone_width = round(stone_width / terrain.horizontal_scale)
    
#     # incline_height = round(incline_height / terrain.vertical_scale)
#     # last_incline_height = round(last_incline_height / terrain.vertical_scale)

#     dis_x = platform_len
#     goals[0] = [platform_len - 1, mid_y]
#     last_dis_x = dis_x
#     for i in range(num_stones):
#         rand_x = np.random.randint(dis_x_min, dis_x_max)
#         rand_y = np.random.randint(dis_y_min, dis_y_max)
#         dis_x += rand_x
#         if not flat:
#             terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
#             terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
#             terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
#         last_dis_x = dis_x
#         goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
#     final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
#     # import ipdb; ipdb.set_trace()
#     if final_dis_x > terrain.width:
#         final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
#     goals[-1] = [final_dis_x, mid_y]
    
#     terrain.goals = goals * terrain.horizontal_scale
    
#     # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
#     # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
#     # terrain.height_field_raw[:, :] = 0
#     # pad edges
#     pad_width = int(pad_width // terrain.horizontal_scale)
#     pad_height = int(pad_height // terrain.vertical_scale)
#     terrain.height_field_raw[:, :pad_width] = pad_height
#     terrain.height_field_raw[:, -pad_width:] = pad_height
#     terrain.height_field_raw[:pad_width, :] = pad_height
#     terrain.height_field_raw[-pad_width:, :] = pad_height

# def parkour_step_terrain(terrain,
#                            platform_len=2.5, 
#                            platform_height=0., 
#                            num_stones=8,
#                             x_range=[0.2, 0.4],
#                            y_range=[-0.15, 0.15],
#                            half_valid_width=[0.45, 0.5],
#                            step_height = 0.2,
#                            pad_width=0.1,
#                            pad_height=0.5):
#     goals = np.zeros((num_stones+2, 2))
#     mid_y = terrain.length // 2  # length is actually y width

#     dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
#     dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
#     dis_y_min = round(y_range[0] / terrain.horizontal_scale)
#     dis_y_max = round(y_range[1] / terrain.horizontal_scale)

#     step_height = round(step_height / terrain.vertical_scale)

#     half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

#     platform_len = round(platform_len / terrain.horizontal_scale)
#     platform_height = round(platform_height / terrain.vertical_scale)
#     terrain.height_field_raw[0:platform_len, :] = platform_height

#     # stone_width = round(stone_width / terrain.horizontal_scale)
    
#     # incline_height = round(incline_height / terrain.vertical_scale)
#     # last_incline_height = round(last_incline_height / terrain.vertical_scale)

#     dis_x = platform_len
#     last_dis_x = dis_x
#     stair_height = 0
#     goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
#     for i in range(num_stones):
#         rand_x = np.random.randint(dis_x_min, dis_x_max)
#         rand_y = np.random.randint(dis_y_min, dis_y_max)
#         if i < num_stones // 2:
#             stair_height += step_height
#         elif i > num_stones // 2:
#             stair_height -= step_height
#         terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
#         dis_x += rand_x
#         terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
#         terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
#         last_dis_x = dis_x
#         goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
#     final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
#     # import ipdb; ipdb.set_trace()
#     if final_dis_x > terrain.width:
#         final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
#     goals[-1] = [final_dis_x, mid_y]
    
#     terrain.goals = goals * terrain.horizontal_scale

#     # pad edges
#     pad_width = int(pad_width // terrain.horizontal_scale)
#     pad_height = int(pad_height // terrain.vertical_scale)
#     terrain.height_field_raw[:, :pad_width] = pad_height
#     terrain.height_field_raw[:, -pad_width:] = pad_height
#     terrain.height_field_raw[:pad_width, :] = pad_height
#     terrain.height_field_raw[-pad_width:, :] = pad_height