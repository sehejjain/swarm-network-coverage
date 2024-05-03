import io
import gymnasium as gym
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np

import PIL


class NetworkEnv(gym.Env):

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
        w1: float = 0.7,
        w2: float = 0.3,
        max_ep_len: int = 1000,
        bonus_threshold: float = 0.9,
        bonus_reward: float = 1000,
        overlap_threshold=0.1,
        penalty_rate=2,
    ):
        super(NetworkEnv, self).__init__()
        self.robots = np.zeros((n, 2), dtype=float)
        self.minX = x[0]
        self.maxX = x[1]
        self.minY = y[0]
        self.maxY = y[1]
        self.n = n
        self.step_size = step
        self.radius = radius
        self.render_mode = render_mode
        self.previous_area_covered = 0
        self.w1 = w1
        self.w2 = w2
        self.max_ep_len = max_ep_len
        self.ep = 0
        self.bonus_threshold = (
            bonus_threshold  # Coverage percentage to reach for the bonus
        )
        self.bonus_reward = bonus_reward  # The reward bonus
        self.total_area = (self.maxX - self.minX) * (self.maxY - self.minY)
        self.overlap_threshold = (
            overlap_threshold  # Area threshold for overlap penalties
        )
        self.penalty_rate = (
            penalty_rate  # Penalty rate per unit area of excessive overlap
        )

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def _observation_space(self):
        return gym.spaces.Dict(
            {
                "robots": gym.spaces.Box(
                    low=np.array([self.minX, self.minY] * self.n).reshape(self.n, 2),
                    high=np.array([self.maxX, self.maxY] * self.n).reshape(self.n, 2),
                    shape=(self.n, 2),
                    dtype=np.float32,
                ),
                "radius": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "minX": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "maxX": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "minY": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "maxY": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "step_size": gym.spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

    def get_obs(self):
        return {
            "robots": self.robots.astype(np.float32),  # Current positions of all drones
            "radius": np.array(
                [self.radius], dtype=np.float32
            ),  # Coverage radius of drones
            "minX": np.array(
                [self.minX], dtype=np.float32
            ),  # Minimum x-coordinate of the environment
            "maxX": np.array(
                [self.maxX], dtype=np.float32
            ),  # Maximum x-coordinate of the environment
            "minY": np.array(
                [self.minY], dtype=np.float32
            ),  # Minimum y-coordinate of the environment
            "maxY": np.array(
                [self.maxY], dtype=np.float32
            ),  # Maximum y-coordinate of the environment
            "step_size": np.array(
                [self.step_size], dtype=np.float32
            ),  # Movement step size
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
                (self.minX, self.minY),
                (self.minX, self.maxY),
                (self.maxX, self.maxY),
                (self.maxX, self.minY),
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
        ax.set_title("Drone Coverage Visualization")
        ax.set_xlabel("X coordinates")
        ax.set_ylabel("Y coordinates")
        # ax.legend()
        ax.grid(True)
        ax.set_xlim(self.minX, self.maxX)
        ax.set_ylim(self.minY, self.maxY)
        ax.set_aspect("equal", adjustable="datalim")
        # ax.axis('off')
        # Display or return the plot based on the rendering mode
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        im = PIL.Image.open(buf)
        plt.close(fig)  # Close the figure to free memory

        if self.render_mode == "human":
            plt.imshow(np.asarray(im))
            plt.axis("off")
            plt.show()
        elif self.render_mode == "rgb_array":
            return np.asarray(im)

    def reset(self, seed=0, options=None):
        self.robots = np.zeros((self.n, 2))
        self.previous_area_covered = 0
        self.ep = 0
        return self.get_obs(), {}

    def step(self, action):
        self.ep += 1
        terminated = False

        reward = 0
        if len(action) != self.n:
            raise ValueError("Action must be of length n")

        for i in range(self.n):
            if action[i] not in self.directions.keys():
                raise ValueError("Invalid action")
            self.robots[i] = self.robots[i] + self.step_size * np.array(
                self.directions[action[i]]
            ) / (np.linalg.norm(self.directions[action[i]]) + 1e-16)
            self.robots[i] = np.clip(
                self.robots[i], [self.minX, self.minY], [self.maxX, self.maxY]
            )
        overlap_penalty = 0
        coverage = [Point(self.robots[i]).buffer(self.radius) for i in range(self.n)]
        union_of_coverage = coverage[0]

        circle_area = coverage[0].area
        for circle in coverage[1:]:
            union_of_coverage = union_of_coverage.union(circle)
            circle_area += circle.area
        for i in range(len(coverage)):
            for j in range(i + 1, len(coverage)):
                if coverage[i].intersects(coverage[j]):
                    overlap_area = coverage[i].intersection(coverage[j]).area
                    # Check if the overlap area exceeds a certain threshold
                    if overlap_area > self.overlap_threshold:
                        overlap_penalty += (
                            overlap_area - self.overlap_threshold
                        ) * self.penalty_rate

        # Check if the coverage is contiguous (single polygon without holes)
        is_contiguous = union_of_coverage.is_simple and union_of_coverage.is_valid

        # Calculate the area of the union and the intersection with the environment boundaries
        area_covered = union_of_coverage.area
        rect = Polygon(
            [
                (self.minX, self.minY),
                (self.minX, self.maxY),
                (self.maxX, self.maxY),
                (self.maxX, self.minY),
            ]
        )
        final_coverage = union_of_coverage.intersection(rect)

        # Define reward
        if not is_contiguous:
            reward = -10  # Penalty for non-contiguous coverage
        else:
            # reward = final_coverage.area -0.1  # Reward is proportional to the covered area

            # Calculate incremental area covered
            incremental_area_covered = area_covered - self.previous_area_covered
            self.previous_area_covered = (
                area_covered  # Update previous area for the next step
            )

            reward = (
                self.w1 * area_covered
                + self.w2 * incremental_area_covered
                - 0.1
                - overlap_penalty
            )  # Reward is a weighted sum of the total and incremental area covered
        coverage_ratio = area_covered / self.total_area
        if self.ep >= self.max_ep_len:
            terminated = True

            if coverage_ratio >= self.bonus_threshold:
                reward += self.bonus_reward

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
