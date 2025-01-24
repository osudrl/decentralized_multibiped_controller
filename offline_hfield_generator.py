import time

import tqdm
import os
import numpy as np
from perlin_noise import PerlinNoise

if __name__ == '__main__':

    os.makedirs('hfields', exist_ok=True)

    while True:
        print('Regenerating hfields...')
        for itr in tqdm.tqdm(range(100)):
            hfield = np.zeros((600, 600), dtype=np.float32)
            perlin_fn = PerlinNoise(octaves=np.random.uniform(5, 10), seed=np.random.randint(10000))

            for i in range(int(600 * 600)):
                ix, iy = i // 600, i % 600

                x = ix * 0.05 - 30 / 2
                y = iy * 0.05 - 30 / 2

                dist = np.linalg.norm((x, y))

                falloff = np.clip((dist - 2) / 2, 0, 1.5)

                z_size = perlin_fn((ix / 600, iy / 600)) * falloff

                hfield[iy, ix] = z_size

            hfield = (hfield - np.min(hfield)) / (np.ptp(hfield) + 1e-6)

            np.save(f'hfields/hfield_{itr}.npy', hfield)


