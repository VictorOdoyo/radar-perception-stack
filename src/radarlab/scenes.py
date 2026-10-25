import numpy as np


def target_from_state(state, amplitude=1):
    from .core import finite
    from .geometry import cartesian_to_polar, radial_measurement
    from .waveform import Target

    x = finite(state, real=True)
    if x.shape != (4,):
        raise ValueError("four-state Cartesian object required")
    r, angle = cartesian_to_polar(x[:2])
    v, _jac = radial_measurement(x)
    return Target(float(r), v, float(angle), amplitude)


def sequence(frames=30, period_s=0.1, model=None, seed=42, drop_indices=(), turn=False):
    from .channel import synthesize
    from .core import integer, positive
    from .waveform import Radar

    integer(seed, 0)
    if seed >= 2**32:
        raise ValueError("32-bit sequence seed required")
    integer(frames, 3)
    period = positive(period_s)
    p = Radar() if model is None else model
    times = np.arange(frames) * period
    initial = np.array([[-4.0, 35.0, 1.2, -1.0], [6.0, 45.0, -0.8, -2.0]])
    truth = np.empty((frames, 2, 4))
    cubes = []
    valid = np.ones(frames, bool)
    for k, t in enumerate(times):
        states = initial.copy()
        states[:, :2] += states[:, 2:] * t
        if turn:
            states[0, 0] += t * t * 0.4
            states[0, 2] += 0.8 * t
        truth[k] = states
        targets = [target_from_state(x, 0.8) for x in states]
        cubes.append(synthesize(targets, p, noise_power=1, seed=(seed + k) % 2**32))
        if k in drop_indices:
            valid[k] = False
    return {"time_s": times, "iq": cubes, "truth": truth, "valid": valid, "model": p}
