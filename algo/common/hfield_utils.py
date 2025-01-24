import glob
import logging

import numpy as np
from shapely import Point, LineString, Polygon, unary_union, box as Box
from shapely.affinity import rotate

from algo.common.utils import normalize

from env.util.quaternion import *


def _get_random_box(env, excluded_space: Polygon, terrain_size, box_size):
    NUM_TRIES = 100
    boundary = Box(-terrain_size[0] / 2, -terrain_size[1] / 2, terrain_size[0] / 2, terrain_size[1] / 2)
    for tries in range(NUM_TRIES):
        x = np.random.uniform(-terrain_size[0] / 2, terrain_size[0] / 2)
        y = np.random.uniform(-terrain_size[1] / 2, terrain_size[1] / 2)
        z = 0
        x_size = np.random.uniform(*box_size[0])
        y_size = np.random.uniform(*box_size[1])
        z_size = np.random.uniform(*box_size[2])
        x_orientation = np.zeros((1,))
        y_orientation = np.zeros((1,))
        z_orientation = np.random.uniform(-np.pi, np.pi, size=(1,))
        if env.hfield_spec['terrain_type'] == 'hfield':
            z_orientation.fill(0)

        hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
        hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

        if hfield_space.within(boundary) and (
                excluded_space is None or not hfield_space.intersects(excluded_space)):
            return x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

    logging.warning(f"Could not find a valid hfield position after {NUM_TRIES} tries.")
    return None


def _get_random_debris(env, excluded_space: Polygon, terrain_size, box_size):
    NUM_TRIES = 100
    boundary = Box(-terrain_size[0] / 2, -terrain_size[1] / 2, terrain_size[0] / 2, terrain_size[1] / 2)
    for tries in range(NUM_TRIES):
        x = np.random.uniform(-terrain_size[0] / 2, terrain_size[0] / 2)
        y = np.random.uniform(-terrain_size[1] / 2, terrain_size[1] / 2)
        z = 0
        x_size = np.random.uniform(*box_size[0])
        y_size = np.random.uniform(*box_size[1])
        z_size = np.random.uniform(*box_size[2])
        x_orientation = np.random.uniform(-np.radians(5), np.radians(5), size=(1,))
        y_orientation = np.random.uniform(-np.radians(5), np.radians(5), size=(1,))
        z_orientation = np.random.uniform(-np.pi, np.pi, size=(1,))
        if env.hfield_spec['terrain_type'] == 'hfield':
            z_orientation.fill(0)

        hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
        hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

        if hfield_space.within(boundary) and (
                excluded_space is None or not hfield_space.intersects(excluded_space)):
            return x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

    logging.warning(f"Could not find a valid hfield position after {NUM_TRIES} tries.")
    return None


def _get_random_balls(env, excluded_space: Polygon, terrain_size, box_size):
    NUM_TRIES = 100
    boundary = Box(-terrain_size[0] / 2, -terrain_size[1] / 2, terrain_size[0] / 2, terrain_size[1] / 2)
    for tries in range(NUM_TRIES):
        x = np.random.uniform(-terrain_size[0] / 2, terrain_size[0] / 2)
        y = np.random.uniform(-terrain_size[1] / 2, terrain_size[1] / 2)
        z = np.random.uniform(1, 10)
        x_size = np.random.uniform(*box_size[0])
        y_size = np.random.uniform(*box_size[1])
        z_size = np.random.uniform(*box_size[2])
        x_orientation, y_orientation, z_orientation = np.zeros((3, 1))

        hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])

        if hfield_space.within(boundary) and (
                excluded_space is None or not hfield_space.intersects(excluded_space)):
            return x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

    logging.warning(f"Could not find a valid hfield position after {NUM_TRIES} tries.")
    return None


def _get_random_walls(env, excluded_space: Polygon, terrain_size, box_size):
    NUM_TRIES = 100
    boundary = Box(-terrain_size[0] / 2, -terrain_size[1] / 2, terrain_size[0] / 2, terrain_size[1] / 2)
    for tries in range(NUM_TRIES):
        x = np.random.uniform(-terrain_size[0] / 2, terrain_size[0] / 2)
        y = np.random.uniform(-terrain_size[1] / 2, terrain_size[1] / 2)
        z = 0
        x_size = np.random.uniform(*box_size[0])
        y_size = np.random.uniform(*box_size[1])
        z_size = np.random.uniform(*box_size[2])
        x_orientation, y_orientation = np.zeros((2, 1))
        z_orientation = np.random.uniform(-np.pi, np.pi, size=(1,))
        if env.hfield_spec['terrain_type'] == 'hfield':
            z_orientation.fill(0)

        hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
        hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

        if hfield_space.within(boundary) and (
                excluded_space is None or not hfield_space.intersects(excluded_space)):
            return x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

    logging.warning(f"Could not find a valid hfield position after {NUM_TRIES} tries.")
    return None


def _fill_terrain_map(env, terrain_map, box):
    terrain_res = env.hfield_spec['terrain_res']
    terrain_size = env.hfield_spec['terrain_size']
    hfield_spacing = env.hfield_spec['hfield_spacing']

    x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation = box

    assert x_orientation.item() == 0 and y_orientation.item() == 0, "Only z orientation is supported."

    box = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
    box = rotate(box, z_orientation, origin='centroid', use_radians=True)

    x_min, y_min, x_max, y_max = box.bounds

    x_min = int(np.clip((x_min + terrain_size[0] / 2) / hfield_spacing[0], 0, terrain_res[0] - 1))
    x_max = int(np.clip((x_max + terrain_size[0] / 2) / hfield_spacing[0], 0, terrain_res[0] - 1))
    y_min = int(np.clip((y_min + terrain_size[1] / 2) / hfield_spacing[1], 0, terrain_res[1] - 1))
    y_max = int(np.clip((y_max + terrain_size[1] / 2) / hfield_spacing[1], 0, terrain_res[1] - 1))

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            x, y = i * hfield_spacing[0] - terrain_size[0] / 2, j * hfield_spacing[1] - terrain_size[1] / 2
            if box.covers(Point(x, y)):
                terrain_map[i, j] = max(terrain_map[i, j], z + z_size)


def get_hfield_terrain_map(env):
    terrain_res = (int(env.hfield_spec['terrain_size'][0] / env.hfield_spec['hfield_spacing'][0]),
                   int(env.hfield_spec['terrain_size'][1] / env.hfield_spec['hfield_spacing'][1]))

    hfield_type = env.hfield_spec['hfield_type']

    match hfield_type:
        case 'random_boxes':
            NUM_BLOCKS = 25

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_BLOCKS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_{i}', 0, 0, -10, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case 'random_debris':
            NUM_BLOCKS = 500

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_BLOCKS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_{i}', 0, 0, -10, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case 'slopy_terrain':
            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(terrain_res[0]):
                for j in range(terrain_res[1]):
                    terrain_map['geom_boxes'].append(
                        (f'geom_box_{i * terrain_res[1] + j}', 0, 0, -10, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1),
                         0, 0, 0))

            return terrain_map

        case 'random_walls':
            NUM_BLOCKS = 30

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_BLOCKS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_{i}', 0, 0, -10, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case 'stairs':
            NUM_GEOMS = 200

            terrain_res = (int(env.hfield_spec['terrain_size'][0] / env.hfield_spec['hfield_spacing'][0]),
                           int(env.hfield_spec['terrain_size'][1] / env.hfield_spec['hfield_spacing'][1]))

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_GEOMS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_{i}', 0, 0, -10, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case 'dynamic_geoms':
            NUM_GEOMS = 10 * 10

            terrain_res = (int(env.hfield_spec['terrain_size'][0] / env.hfield_spec['hfield_spacing'][0]),
                           int(env.hfield_spec['terrain_size'][1] / env.hfield_spec['hfield_spacing'][1]))

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_GEOMS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_NC_{i}', 0, 0, 0, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case 'bumps':
            NUM_GEOMS = 10

            terrain_res = (int(env.hfield_spec['terrain_size'][0] / env.hfield_spec['hfield_spacing'][0]),
                           int(env.hfield_spec['terrain_size'][1] / env.hfield_spec['hfield_spacing'][1]))

            terrain_map = {'geom_boxes': [],
                           'geom_heights': np.zeros(terrain_res, dtype=np.float32)}

            for i in range(NUM_GEOMS):
                terrain_map['geom_boxes'].append(
                    (f'geom_box_{i}', 0, 0, 0, 0.1, 0.1, 0.1, (0.5, 0.5, 0.5, 1), 0, 0, 0))

            return terrain_map

        case _:
            raise ValueError(f"Invalid hfield type {hfield_type}")


def randomize_hfield(env):
    base_positions = env.get_base_position()[:, :2]

    robot_space = Point(base_positions[0, :2]).buffer(0.25)

    for pos in base_positions[1:]:
        robot_space = robot_space.union(Point(pos).buffer(0.25))
    # match env.num_cassie:
    #     case 1:
    #         robot_space = Point(base_positions[0, :2])  # .buffer(0.5)
    #
    #         for pos in base_positions[1:]:
    #             robot_space = robot_space.union(Point(pos).buffer(0.5))
    #     case 2:
    #         robot_space = LineString(base_positions[:, :2])  # .buffer(0.5)
    #     case _:
    #         robot_space = Polygon(base_positions)  # .buffer(0.5)

    # robot_space = robot_space.convex_hull.buffer(0.5)

    hfield_type = env.hfield_spec['hfield_type']

    env.terrain_map['geom_heights'] = np.zeros(env.hfield_spec['terrain_res'], dtype=np.float32)

    if env.hfield_spec['terrain_type'] == 'hfield':
        hfield = env.sim.model.hfield('hfield0').data
        hfield.fill(0)
        hfield_radius_x, hfield_radius_y, hfield_max_z, hfield_min_z = env.sim.model.hfield('hfield0').size

    terrain_res = env.hfield_spec['terrain_res']

    if hfield_type == 'random_boxes':
        for i, (geom_name, x, y, z, x_size, y_size, z_size, rgba, x_orientation, y_orientation, z_orientation) \
                in enumerate(env.terrain_map['geom_boxes']):

            box = _get_random_box(env,
                                  excluded_space=robot_space,
                                  terrain_size=env.hfield_spec['terrain_size'],
                                  box_size=((0.5, 2), (0.5, 2), (0.05, 0.3)))

            if box is None:
                continue

            x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation = box
            # z_orientation = euler[0]

            # Update the terrain map
            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'hfield':
                # hfield[x:x + x_size, y:y + y_size] = z / hfield_max_z

                x = int(normalize(x, src_min=-env.hfield_spec['terrain_size'][0] / 2,
                                  src_max=env.hfield_spec['terrain_size'][0] / 2,
                                  trg_min=0, trg_max=hfield.shape[0] - 1))

                y = int(normalize(y, src_min=-env.hfield_spec['terrain_size'][1] / 2,
                                  src_max=env.hfield_spec['terrain_size'][1] / 2,
                                  trg_min=0, trg_max=hfield.shape[1] - 1))

                x_size = int(normalize(x_size, src_min=0, src_max=env.hfield_spec['terrain_size'][0],
                                       trg_min=0, trg_max=hfield.shape[0] - 1))
                y_size = int(normalize(y_size, src_min=0, src_max=env.hfield_spec['terrain_size'][1],
                                       trg_min=0, trg_max=hfield.shape[1] - 1))

                hfield[y:y + y_size, x:x + x_size] = z_size / hfield_max_z
            else:
                # Update the geoms
                env.sim.set_geom_pose(geom_name,
                                      [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                       *euler2quat(z=z_orientation, y=y_orientation, x=x_orientation)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    elif hfield_type == 'random_debris':
        for i, (geom_name, x, y, z, x_size, y_size, z_size, rgba, x_orientation, y_orientation, z_orientation) \
                in enumerate(env.terrain_map['geom_boxes']):

            box = _get_random_debris(env,
                                     excluded_space=robot_space,
                                     terrain_size=env.hfield_spec['terrain_size'],
                                     box_size=((0.1, 0.6), (0.5, 2), (0.05, 0.25)))

            if box is None:
                continue

            x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation = box

            # Update the terrain map
            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'hfield':
                # hfield[x:x + x_size, y:y + y_size] = z / hfield_max_z

                x = int(normalize(x, src_min=-env.hfield_spec['terrain_size'][0] / 2,
                                  src_max=env.hfield_spec['terrain_size'][0] / 2,
                                  trg_min=0, trg_max=hfield.shape[0] - 1))

                y = int(normalize(y, src_min=-env.hfield_spec['terrain_size'][1] / 2,
                                  src_max=env.hfield_spec['terrain_size'][1] / 2,
                                  trg_min=0, trg_max=hfield.shape[1] - 1))

                x_size = int(normalize(x_size, src_min=0, src_max=env.hfield_spec['terrain_size'][0],
                                       trg_min=0, trg_max=hfield.shape[0] - 1))
                y_size = int(normalize(y_size, src_min=0, src_max=env.hfield_spec['terrain_size'][1],
                                       trg_min=0, trg_max=hfield.shape[1] - 1))

                hfield[y:y + y_size, x:x + x_size] = z_size / hfield_max_z
            else:
                # Update the geoms
                env.sim.set_geom_pose(geom_name,
                                      [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                       *euler2quat(z=z_orientation, y=y_orientation, x=x_orientation)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            # Unsupported
            # _fill_terrain_map(env.terrain_map['geom_heights'], box)

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    elif hfield_type == 'random_walls':
        for i, (geom_name, x, y, z, x_size, y_size, z_size, rgba, x_orientation, y_orientation, z_orientation) \
                in enumerate(env.terrain_map['geom_boxes']):

            box = _get_random_walls(env,
                                    excluded_space=unary_union([robot_space, Point(7, 7).buffer(1.0)]),
                                    terrain_size=env.hfield_spec['terrain_size'],
                                    box_size=((0.1, 0.6), (0.5, 1.5), (0.5, 2)))

            if box is None:
                continue

            x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation = box

            # Update the terrain map
            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'hfield':
                # hfield[x:x + x_size, y:y + y_size] = z / hfield_max_z

                x = int(normalize(x, src_min=-env.hfield_spec['terrain_size'][0] / 2,
                                  src_max=env.hfield_spec['terrain_size'][0] / 2,
                                  trg_min=0, trg_max=hfield.shape[0] - 1))

                y = int(normalize(y, src_min=-env.hfield_spec['terrain_size'][1] / 2,
                                  src_max=env.hfield_spec['terrain_size'][1] / 2,
                                  trg_min=0, trg_max=hfield.shape[1] - 1))

                x_size = int(normalize(x_size, src_min=0, src_max=env.hfield_spec['terrain_size'][0],
                                       trg_min=0, trg_max=hfield.shape[0] - 1))
                y_size = int(normalize(y_size, src_min=0, src_max=env.hfield_spec['terrain_size'][1],
                                       trg_min=0, trg_max=hfield.shape[1] - 1))

                hfield[y:y + y_size, x:x + x_size] = z_size / hfield_max_z
            else:
                # Update the geoms
                env.sim.set_geom_pose(geom_name,
                                      [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                       *euler2quat(z=z_orientation, y=y_orientation, x=x_orientation)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    elif hfield_type == 'slopy_terrain':
        # pre load to save compute time
        hfield_paths = glob.glob('hfields/*.npy')
        hfield_path = hfield_paths[np.random.randint(0, len(hfield_paths))]

        try:
            hfield[:] = np.load(hfield_path)
        except Exception as e:
            logging.error(f'Error loading hfield {hfield_path}. {e}')

        # perlin_fn = PerlinNoise(octaves=np.random.uniform(5, 10), seed=np.random.randint(10000))
        #
        # for i, _ in enumerate(env.terrain_map['geom_boxes']):
        #
        #     ix, iy = i // terrain_res[1], i % terrain_res[1]
        #
        #     x = ix * env.hfield_spec['hfield_spacing'][0] - env.hfield_spec['terrain_size'][0] / 2
        #     y = iy * env.hfield_spec['hfield_spacing'][1] - env.hfield_spec['terrain_size'][1] / 2
        #
        #     if robot_space.contains(Point(x, y)):
        #         falloff = 0
        #     else:
        #         falloff = normalize(src=robot_space.exterior.distance(Point(x, y)),
        #                             src_min=0, src_max=2, trg_min=0, trg_max=1)
        #
        #     z_size = perlin_fn((ix / terrain_res[0], iy / terrain_res[1])) * falloff
        #
        #     if env.hfield_spec['terrain_type'] == 'hfield':
        #         hfield[iy, ix] = z_size
        #     else:
        #         raise ValueError(f"Invalid terrain type {env.hfield_spec['terrain_type']} for hfield")

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    elif hfield_type == 'stairs':
        num_boxes = len(env.terrain_map['geom_boxes'])

        xmin, ymin, xmax, ymax = robot_space.buffer(0.1).bounds

        max_height = 1.2

        x = xmax
        z = 0

        x_size = 0
        z_size = 0

        direction = 'up'

        boxes = env.terrain_map['geom_boxes']  [:num_boxes // 4]

        for i, (geom_name, *_) in enumerate(boxes):

            x += x_size
            y = -10

            if z + z_size >= max_height:
                direction = 'down'

            if direction == 'up':
                z += z_size

            x_size = np.random.uniform(0.3, 0.6)

            y_size = 20
            z_size = np.random.uniform(0.05, 0.2)

            if z - z_size < 0:
                direction = 'up'

            if direction == 'down':
                z -= z_size

            x_orientation, y_orientation, z_orientation = np.zeros((3, 1), dtype=np.float32)

            box = x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

            hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
            hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

            if robot_space is not None and hfield_space.intersects(robot_space):
                continue

            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'geom':
                # Update the geoms
                env.sim.set_geom_pose(geom_name, [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                                  *euler2quat(z=z_orientation, y=0, x=0)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        x = xmin
        z = 0

        x_size = 0
        z_size = 0

        direction = 'up'

        boxes = env.terrain_map['geom_boxes'][num_boxes // 4 + 1:num_boxes // 2]

        for i, (geom_name, *_) in enumerate(boxes):

            y = -10

            if z + z_size >= max_height:
                direction = 'down'

            if direction == 'up':
                z += z_size

            x_size = np.random.uniform(0.3, 0.6)
            y_size = 20
            z_size = np.random.uniform(0.05, 0.2)

            if z - z_size < 0:
                direction = 'up'

            if direction == 'down':
                z -= z_size

            x -= x_size

            x_orientation, y_orientation, z_orientation = np.zeros((3, 1), dtype=np.float32)

            box = x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

            hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
            hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

            if robot_space is not None and hfield_space.intersects(robot_space):
                continue

            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'geom':
                # Update the geoms
                env.sim.set_geom_pose(geom_name, [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                                  *euler2quat(z=z_orientation, y=0, x=0)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        y = ymax
        z = 0

        y_size = 0
        z_size = 0

        direction = 'up'

        boxes = env.terrain_map['geom_boxes'][num_boxes // 2 + 1:num_boxes * 3 // 4]

        for i, (geom_name, *_) in enumerate(boxes):

            x = -10
            y += y_size

            if z + z_size >= max_height:
                direction = 'down'

            if direction == 'up':
                z += z_size

            y_size = np.random.uniform(0.2, 0.6)
            x_size = 20
            z_size = np.random.uniform(0.05, 0.2)

            if z - z_size < 0:
                direction = 'up'

            if direction == 'down':
                z -= z_size

            x_orientation, y_orientation, z_orientation = np.zeros((3, 1), dtype=np.float32)

            box = x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

            hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
            hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

            if robot_space is not None and hfield_space.intersects(robot_space):
                continue

            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'geom':
                # Update the geoms
                env.sim.set_geom_pose(geom_name, [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                                  *euler2quat(z=z_orientation, y=0, x=0)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        y = ymin
        z = 0

        y_size = 0
        z_size = 0

        direction = 'up'

        boxes = env.terrain_map['geom_boxes'][num_boxes * 3 // 4 + 1:]

        for i, (geom_name, *_) in enumerate(boxes):

            x = -10

            if z + z_size >= max_height:
                direction = 'down'

            if direction == 'up':
                z += z_size

            y_size = np.random.uniform(0.2, 0.6)
            x_size = 20
            z_size = np.random.uniform(0.05, 0.2)

            if z - z_size < 0:
                direction = 'up'

            if direction == 'down':
                z -= z_size

            y -= y_size

            x_orientation, y_orientation, z_orientation = np.zeros((3, 1), dtype=np.float32)

            box = x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

            hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
            hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

            if robot_space is not None and hfield_space.intersects(robot_space):
                continue

            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'geom':
                # Update the geoms
                env.sim.set_geom_pose(geom_name, [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                                  *euler2quat(z=z_orientation, y=0, x=0)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    elif hfield_type == 'dynamic_geoms':
        num_bodies_after_cassies = len(env.terrain_map['geom_boxes'])

        qpos_boxes = np.full((num_bodies_after_cassies, 7), 1.0)

        terrain_size = env.hfield_spec['terrain_size']

        grid_size = 10, 10

        assert np.prod(grid_size) <= num_bodies_after_cassies

        xs, ys = np.meshgrid(
            np.linspace(-terrain_size[0] / 2, terrain_size[0] / 2, grid_size[0]),
            np.linspace(-terrain_size[1] / 2, terrain_size[1] / 2, grid_size[1])
        )

        idx = np.random.permutation(num_bodies_after_cassies)

        for i, (geom_name, x, y, z, x_size, y_size, z_size, rgba, x_orientation, y_orientation, z_orientation) \
                in enumerate(env.terrain_map['geom_boxes']):
            # box = _get_random_balls(env,
            #                         excluded_space=robot_space,
            #                         terrain_size=env.hfield_spec['terrain_size'],
            #                         box_size=((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))

            x_orientation, y_orientation, z_orientation = np.random.uniform(-np.pi, np.pi, (3, 1))
            # y_orientation = np.full((1,), np.pi / 2)
            # y_orientation = np.zeros((1,))
            # x_orientation = np.random.uniform(-np.pi, np.pi, (1,))
            # z_orientation = np.zeros((1,))

            x_size, y_size = np.random.uniform(0.1, 0.15), np.random.uniform(0.3, 0.5)

            box = (xs[i // grid_size[0]][i % grid_size[1]], ys[i // grid_size[0]][i % grid_size[1]], 0.3,
                   x_size, y_size, 0.1,
                   x_orientation, y_orientation, z_orientation)
            # if box is None:
            #     continue

            x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation = box

            # Update the terrain map
            # env.terrain_map['geom_boxes'][idx[i]] = (f'geom_box_NC_{i}',
            #                                     x, y, z,
            #                                     x_size, y_size, z_size,
            #                                     (0.5, 0.5, 0.5, 1),
            #                                     x_orientation, y_orientation, z_orientation)

            qpos_boxes[idx[i]] = [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                  *euler2quat(z=z_orientation, y=y_orientation, x=x_orientation)[0]]

            env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

        return qpos_boxes

    elif hfield_type == 'bumps':
        xmin, ymin, xmax, ymax = robot_space.buffer(0.1).bounds

        x = xmax
        z = 0

        x_size = 0

        boxes = env.terrain_map['geom_boxes']

        for i, (geom_name, *_) in enumerate(boxes):

            x += x_size + np.random.uniform(0.5, 1.0)
            y = -10

            x_size = np.random.uniform(0.3, 0.6)

            y_size = 20

            z_size = np.random.uniform(0.1, 0.3)

            x_orientation, y_orientation, z_orientation = np.zeros((3, 1), dtype=np.float32)

            box = x, y, z, x_size, y_size, z_size, x_orientation, y_orientation, z_orientation

            hfield_space = Polygon([(x, y), (x, y + y_size), (x + x_size, y + y_size), (x + x_size, y)])
            hfield_space = rotate(hfield_space, z_orientation, origin='centroid', use_radians=True)

            if robot_space is not None and hfield_space.intersects(robot_space):
                continue

            env.terrain_map['geom_boxes'][i] = (f'geom_box_{i}',
                                                x, y, z,
                                                x_size, y_size, z_size,
                                                (0.5, 0.5, 0.5, 1),
                                                x_orientation, y_orientation, z_orientation)

            if env.hfield_spec['terrain_type'] == 'geom':
                # Update the geoms
                env.sim.set_geom_pose(geom_name, [x + x_size / 2, y + y_size / 2, z + z_size / 2,
                                                  *euler2quat(z=z_orientation, y=0, x=0)[0]])
                env.sim.set_geom_size(geom_name, [x_size / 2, y_size / 2, z_size / 2])

            _fill_terrain_map(env, env.terrain_map['geom_heights'], box)

        if env.terrain_map['terrain_type'] == 'hfield':
            env.sim.upload_hfield()

    else:
        raise ValueError(f"Invalid hfield type {env.hfield_spec['hfield_type']}")
