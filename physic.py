import numpy as np


def bord_reflex(coord, bord, speed, bord_speed, radius):
    mask = coord[:, np.newaxis, :] - bord[np.newaxis, :, :]
    mask[:, 1, :] *= -1
    mask = mask <= radius
    speed[mask[:, 0, :] | mask[:, 1, :]] *= -1
    speed[mask[:, 0, 0], 0] += 2 * bord_speed
    coord = np.maximum(bord[0] + radius,
                       np.minimum(coord, bord[1] - radius))
    return coord, speed


def norm(x):
    return x / np.sqrt(np.sum(x ** 2, axis=1)).reshape(-1, 1)


def mol_reflex(coord, speed, eld_mask, radius):
    dist = np.sqrt(np.sum((coord[:, np.newaxis, :] -
                           coord[np.newaxis, :, :]) ** 2, axis=2))
    mask = (dist > 0) & (dist <= 2 * radius)
    dif_speed = speed[np.newaxis, :, :] - speed[:, np.newaxis, :]
    direction = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
    mask = mask & (np.sum(dif_speed * direction, axis=2) > 0)
    ret_mask = mask.copy()
    mask = ~eld_mask & mask
    if not np.any(mask):
        return speed, ret_mask
    r = norm(direction[mask])
    dif_speed[mask] = np.abs(np.sum(dif_speed[mask] * r,
                                    axis=1))[:, np.newaxis] * r
    new_speed = dif_speed + speed[:, np.newaxis, :]
    new_speed[~mask] = np.nan
    speed[np.any(mask, axis=1)] = \
        np.nanmean(new_speed, axis=1)[np.any(mask, axis=1)]
    return speed, ret_mask


def iteration(coord, speed, dt, mask, bord_box, bord_speed, radius):
    coord += speed * dt
    coord, speed = bord_reflex(coord, bord_box, speed, bord_speed,
                               radius)
    speed, mask = mol_reflex(coord, speed, mask, radius)
    return coord, speed, mask


R = 8.3144598


def temperature(speed, N, mass, size_coef):
    mean_speed = np.mean(np.sum(speed ** 2, axis=1)) * size_coef ** 2
    return mean_speed * mass * N / R / 2
