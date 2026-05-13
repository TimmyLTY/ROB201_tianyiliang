from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.gui_map.closed_playground import ClosedPlayground
from place_bot.simulation.gui_map.world_abstract import WorldAbstract

from worlds import walls_my_world


class MyWorld(WorldAbstract):

    def __init__(self, robot: RobotAbstract, use_shaders: bool = True):
        super().__init__(robot=robot)

        self._size_area = (1113, 750)

        self._playground = ClosedPlayground(size=self._size_area, use_shaders=use_shaders)
        walls_my_world.add_walls(self._playground)
        walls_my_world.add_boxes(self._playground)

        angle = 0
        self._robot_pos = ((439.0, 195), angle)
        self._playground.add(robot, self._robot_pos)
