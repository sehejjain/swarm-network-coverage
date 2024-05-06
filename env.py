"""Network Environment"""

import io
import gymnasium as gym
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np

import PIL


class NetworkEnv(gym.Env):
    """Class for Environment for Network Coverage Problem."""

    directions = {
        0: (0, 0),  # No action
        1: (0, 1),  # North
        2: (0, -1),  # South
        3: (1, 0),  # East
        4: (-1, 0),  # West
        5: (1, 1),  # Northeast
        6: (-1, 1),  # Northwest
        7: (1, -1),  # Southeast
        8: (-1, -1),  # Southwest
    }

    def __init__(
        self,
        n: int,
        x: tuple,
        y: tuple,
        step: float = 1.0,
        radius: float = 1.0,
        render_mode: str = "rgb_array",
        w1: float = 1,
        w2: float = 2,
        w3: float = 0.1,
        max_ep_len: int = 1000,
        bonus_threshold: float = 0.9,
        bonus_reward: float = 1000,
        overlap_threshold=0.1,
        train: bool = True,
    ):
        """
        Initializes the NetworkEnv object.

        Parameters:
        - n (int): Number of robots.
        - x (tuple): Tuple containing the minimum and maximum x-coordinates of the environment.
        - y (tuple): Tuple containing the minimum and maximum y-coordinates of the environment.
        - step (float, optional): Step size for robot movement. Defaults to 1.0.
        - radius (float, optional): Radius of the robots. Defaults to 1.0.
        - render_mode (str, optional): Render mode for visualization. Defaults to "rgb_array".
        - w1 (float, optional): Weight for the coverage. Defaults to 0.7.
        - w2 (float, optional): Weight for the coverage change. Defaults to 0.3.
        - w3 (float, optional): Weight for the overlap penalty. Defaults to 0.1.
        - max_ep_len (int, optional): Maximum episode length. Defaults to 1000.
        - bonus_threshold (float, optional): Coverage percentage for the bonus. Defaults to 0.9.
        - bonus_reward (float, optional): The reward bonus. Defaults to 1000.
        - overlap_threshold (float, optional): Area threshold for overlap. Defaults to 0.1.
        """
        super(NetworkEnv, self).__init__()
        self.train=train
        self.robots = np.zeros((n, 2), dtype=float) if not train else np.random.randint(0, x[1], (n, 2))
        self.min_x = x[0]
        self.max_x = x[1]
        self.min_y = y[0]
        self.max_y = y[1]
        self.n = n
        self.step_size = step
        self.radius = radius
        self.render_mode = render_mode
        self.previous_area_covered = 0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.max_ep_len = max_ep_len

        self.bonus_threshold = bonus_threshold
        self.bonus_reward = bonus_reward  # The reward bonus
        self.total_area = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        self.overlap_threshold = overlap_threshold

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.ep = 0
        self.coverage_ratio = 0

    def _observation_space(self):
        return gym.spaces.Dict(
            {
                "robots": gym.spaces.Box(
                    low=np.array([self.min_x, self.min_y] * self.n).reshape(self.n, 2),
                    high=np.array([self.max_x, self.max_y] * self.n).reshape(self.n, 2),
                    shape=(self.n, 2),
                    dtype=np.float32,
                ),
                "radius": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "min_x": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "max_x": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "min_y": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "max_y": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "step_size": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "coverage_ratio": gym.spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),
            }
        )

    def get_obs(self):
        """
        Get the observation of the environment.

        Returns:
            dict: A dictionary containing the following information:
                - "robots": Current positions of all drones as a numpy array
                - "radius": Coverage radius of drones as a numpy array
                - "min_x": Minimum x-coordinate of the environment as a numpy array
                - "max_x": Maximum x-coordinate of the environment as a numpy array
                - "min_y": Minimum y-coordinate of the environment as a numpy array
                - "max_y": Maximum y-coordinate of the environment as a numpy array
                - "step_size": Movement step size as a numpy array
        """
        return {
            "robots": self.robots.astype(np.float32),  # Current positions of all drones
            "radius": np.array(
                [self.radius], dtype=np.float32
            ),  # Coverage radius of drones
            "min_x": np.array(
                [self.min_x], dtype=np.float32
            ),  # Minimum x-coordinate of the environment
            "max_x": np.array(
                [self.max_x], dtype=np.float32
            ),  # Maximum x-coordinate of the environment
            "min_y": np.array(
                [self.min_y], dtype=np.float32
            ),  # Minimum y-coordinate of the environment
            "max_y": np.array(
                [self.max_y], dtype=np.float32
            ),  # Maximum y-coordinate of the environment
            "step_size": np.array(
                [self.step_size], dtype=np.float32
            ),  # Movement step size
            "coverage_ratio": np.array(
                [self.coverage_ratio], dtype=np.float32
            ),  # Coverage ratio
        }

    def _action_space(self):
        return gym.spaces.MultiDiscrete([9] * self.n)

    def render(self):
        coverage = [Point(self.robots[i]).buffer(self.radius) for i in range(self.n)]

        union_of_coverage = coverage[0]
        for circle in coverage[1:]:
            union_of_coverage = union_of_coverage.union(circle)

        # Calculate the area of the union

        rect = Polygon(
            [
                (self.min_x, self.min_y),
                (self.min_x, self.max_y),
                (self.max_x, self.max_y),
                (self.max_x, self.min_y),
            ]
        )
        final = union_of_coverage.intersection(rect)

        fig, ax = plt.subplots()

        # Define a function to plot polygons with centroids
        def plot_polygon(ax, polygon):
            if isinstance(polygon, Polygon):
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.3, fc="skyblue", edgecolor="navy", linewidth=2)

                for interior in polygon.interiors:
                    ix, iy = interior.xy
                    ax.fill(ix, iy, alpha=1, fc="white", edgecolor="black")

                # Plot the centroids of the coverage areas
                # centroid = polygon.centroid
                # ax.plot(centroid.x, centroid.y, 'o', markersize=10, color='red', label='Centroid')

            else:
                raise ValueError("Input must be a shapely Polygon")

        # Plot each polygon in the MultiPolygon
        if not isinstance(final, Polygon):
            for polygon in final.geoms:
                plot_polygon(ax, polygon)
        else:
            plot_polygon(ax, final)

        # Plot robot positions
        for robot in self.robots:
            ax.plot(
                robot[0],
                robot[1],
                "p",
                markersize=12,
                color="gold",
                label="Robot Position",
            )

        # Enhance the plot with titles, labels, and a legend
        # ax.legend()
        ax.grid(True)
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)
        # ax.set_aspect("equal", adjustable="datalim")
        # ax.axis('off')
        # Display or return the plot based on the rendering mode
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100,bbox_inches='tight')
        buf.seek(0)
        im = PIL.Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        if self.render_mode == "human":
            plt.imshow(np.asarray(im))
            plt.axis("off")
            plt.show()
        elif self.render_mode == "rgb_array":
            return np.asarray(im)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        
        self.robots = np.zeros((self.n, 2), dtype=float) if not self.train else np.random.randint(0, self.max_x, (self.n, 2))
        self.previous_area_covered = 0
        self.ep = 0
        self.coverage_ratio = 0
        return self.get_obs(), {}


    def get_reward(self):
        terminated = False
        reward = 0
        overlap_penalty = 0
        coverage = [Point(self.robots[i]).buffer(self.radius) for i in range(self.n)]
        union_of_coverage = coverage[0]

        circle_area = coverage[0].area
        for circle in coverage[1:]:
            union_of_coverage = union_of_coverage.union(circle)
            circle_area += circle.area
        for i, coverage_i in enumerate(coverage):
            for coverage_j in coverage[i + 1 :]:
                if coverage_i.intersects(coverage_j):
                    overlap_area = coverage_i.intersection(coverage_j).area
                    # Check if the overlap area exceeds a certain threshold
                    if overlap_area / coverage_i.area > self.overlap_threshold:
                        overlap_penalty += overlap_area / coverage_i.area

        # Check if the coverage is contiguous (single polygon without holes)
        if union_of_coverage.geom_type == "MultiPolygon":
            if len(list(union_of_coverage.geoms))!=0:
                is_contiguous = False
        else:
            is_contiguous = True

        # Calculate the area of the union and the intersection with the environment boundaries

        rect = Polygon(
            [
                (self.min_x, self.min_y),
                (self.min_x, self.max_y),
                (self.max_x, self.max_y),
                (self.max_x, self.min_y),
            ]
        )
        final_coverage = union_of_coverage.intersection(rect)
        area_covered = final_coverage.area

        coverage_ratio = area_covered / self.total_area

        self.coverage_ratio = coverage_ratio

        # Define reward
        if not is_contiguous:
            reward = -10  # Penalty for non-contiguous coverage
        else:
            increase_in_coverage_ratio = (
                coverage_ratio - self.previous_area_covered / self.total_area
            )

            self.previous_area_covered = area_covered

            reward = (
                self.w1 * coverage_ratio
                + self.w2 * increase_in_coverage_ratio
                - 0.1
                - self.w3 * overlap_penalty
            )  # Reward is a weighted sum of the total and incremental area covered

        if self.ep >= self.max_ep_len:
            terminated = True

        if coverage_ratio >= self.bonus_threshold and is_contiguous:
            reward += self.bonus_reward
                
                
        return reward, terminated, final_coverage, is_contiguous, coverage_ratio
        
        
        
    def step(self, action):
        self.ep += 1
        

        reward = 0
        if len(action) != self.n:
            raise ValueError("Action must be of length n")

        for i in range(self.n):
            if action[i] not in self.directions:
                raise ValueError("Invalid action")
            self.robots[i] = self.robots[i] + self.step_size * np.array(
                self.directions[action[i]]
            ) / (np.linalg.norm(self.directions[action[i]]) + 1e-16)
            self.robots[i] = np.clip(
                self.robots[i], [self.min_x, self.min_y], [self.max_x, self.max_y]
            )
        
        reward, terminated, final_coverage, is_contiguous, coverage_ratio = self.get_reward()
        return (
            self.get_obs(),
            reward,
            terminated,
            False,
            {
                "area_covered": final_coverage.area,
                "is_contiguous": is_contiguous,
                "coverage_ratio": coverage_ratio,
            },
        )
