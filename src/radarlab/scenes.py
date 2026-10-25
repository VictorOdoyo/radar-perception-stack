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
