import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from constants import dtype


class AdSBHNet(nn.Module):
    def __init__(self, N=2, std=1.0):
        super(AdSBHNet, self).__init__()
        self.a = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dtype))
        self.b = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dtype))
        '''
        `self.logcoef` is the log of the dimensionless parameter
        R^2/(2*pi*alpha') which multiplies the static potential V.
        '''
        self.logcoef = nn.Parameter(
            torch.normal(0.0, std, size=(1,), dtype=dtype)[0]
        )
        '''
        The lattice data is supposed to be shifted such that it behaves
        correctly in the UV. `self.shift` holds that parameter.
        '''
        self.shift = nn.Parameter(
            torch.normal(0.0, std, size=(1,), dtype=dtype)[0]
        )

    def forward(self, Ls):
        return predict(self.a, self.b, self.logcoef, self.shift, Ls)

    def integrate_L(self, z):
        return integrate_L(self.a, self.b, z)

    def integrate_V(self, z):
        coef = self.logcoef.exp()
        return coef * integrate_V(self.a, self.b, z) + self.shift

    def eval_f(self, z):
        return eval_f(self.a, z)

    def eval_g(self, z):
        return eval_g(self.b, z)


def check_positive_a(a):
    '''
    Checks that the function a(z) defined by self.a
    is positive in the range [0, 1]. We'll do this in
    an inelegant way because a(z) can be a high order
    polynomial where we don't know the minima analytically.
    '''
    with torch.no_grad():
        z_grid = torch.linspace(0, 0.999, steps=1000)
        _a = torch.cat((torch.tensor([1.0]), a))
        a_grid = torch.zeros_like(z_grid)
        for i, ci in enumerate(_a):
            a_grid += ci * z_grid**i
        assert torch.all(a_grid > 0), (
            f'a(z) is not positive for a={a}.'
        )


def check_positive_b(b):
    '''
    Same for b(z).
    '''
    with torch.no_grad():
        z_grid = torch.linspace(0, 0.999, steps=1000)
        _b = torch.cat((torch.tensor([1.0]), b))
        b_grid = torch.zeros_like(z_grid)
        for i, ci in enumerate(_b):
            b_grid += ci * z_grid**i
        assert torch.all(b_grid > 0), (
            f'b(z) is not positive for b={b}.'
        )


def as_tensor(tensor):
    '''
    Converts `tensor` to a torch.tensor if it
    is not already a tensor.
    '''
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return torch.as_tensor(tensor, dtype=dtype)


def predict(a, b, logcoef, shift, Ls):
    '''
    Computes the potentials corresponding to the quark separations `Ls`
    in the holographic model defined by `a`, `b`, `logcoef`, and `shift`.
    '''
    # Check that inputs are tensors
    a = as_tensor(a)
    b = as_tensor(b)
    logcoef = as_tensor(logcoef)
    shift = as_tensor(shift)

    # Check that a(z) > 0 and b(z) > 0
    check_positive_a(a)
    check_positive_b(b)

    V = torch.zeros_like(Ls, dtype=dtype)

    zs_max, L_max = get_L_max(a, b)
    curve = interp1d([0.0, L_max.item()], [0.0, zs_max.item()])
    zs_crit, L_crit = get_L_crit(a, b, zs_max)

    for i, L in enumerate(Ls):
        if L < L_crit:
            # String is not broken
            init = curve(L.item())
            zs = find_zs_newton(a, b, L, init)
            # zs = find_zs_binary(a, b, L, zs_max)
            assert zs > 0, f'zs is negative: {zs} for L = {L}'
            V[i] = integrate_V(a, b, zs)
        else:
            # String is broken
            V[i] = as_tensor(0.0)
        assert not torch.isnan(V[i])
    assert not torch.all(V == 0.0), (
        'String is broken for all Ls.'
        'The loss might not have a gradient.'
    )
    coef = logcoef.exp()
    V = coef * V + shift
    return V


def find_zs_newton(a, b, L, init, max_steps=25):
    init = as_tensor(init)
    zs = [init]
    _L = integrate_L(a, b, zs[-1])
    for i in range(max_steps):
        dL = integrate_dL(a, b, zs[-1])
        zs.append(zs[-1] - (_L - L) / dL)
        assert 0 < zs[-1] < 1 and not torch.isnan(zs[-1]), (
            f'Something wrong in Newton:\n\tzs = {[_zs.item() for _zs in zs]}'
            '\n\tL = {L}\n\t_L = {_L}\n\tdL = {dL}'
        )
        _L = integrate_L(a, b, zs[-1])
        diff = torch.abs(_L - L)
        if diff < 1e-8:
            return zs[-1]
    # Max steps was reached with no convergence
    assert False, (
        f'Newton\'s method failed to converge in {max_steps} iterations for'
        'L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}.'
    )


def find_zs_binary(a, b, L, zs_max):
    '''
    Intended for real zs.
    '''
    zs_low = 0.001
    zs_high = zs_max
    L_low = integrate_L(a, b, zs_low)
    L_high = integrate_L(a, b, zs_high)
    while zs_high - zs_low > 1e-8:
        zs_mid = (zs_high + zs_low) / 2
        L_mid = integrate_L(a, b, zs_mid)
        L_mid = L_mid
        if L_mid < L:
            zs_low = zs_mid
            L_low = L_mid
        else:
            zs_high = zs_mid
            L_high = zs_mid
    # zs_mid = (zs_high + zs_low) / 2
    zs_mid = zs_low + (zs_high - zs_low) / (L_high - L_low) * (L - L_low)
    return zs_mid


def get_L_max(a, b):
    '''
    Returns the point where L is maximal such that
    zs is still real. This is the last point on the
    real axis along the real L curve.
    '''
    zs_UV, zs_IR = 0.001, 0.999
    with torch.no_grad():
        dL_IR = integrate_dL(a, b, zs_IR)
        dL_UV = integrate_dL(a, b, zs_UV)
        assert dL_IR < 0 and dL_UV > 0
        while zs_IR - zs_UV > 1e-8:
            zs_mid = (zs_UV + zs_IR) / 2
            dL_mid = integrate_dL(a, b, zs_mid)
            if dL_mid < 0:
                zs_IR = zs_mid
            else:
                zs_UV = zs_mid
        zs_mid = (zs_UV + zs_IR) / 2
        L_max = integrate_L(a, b, zs_mid)
    zs_mid = torch.tensor(zs_mid, dtype=dtype)
    return zs_mid, L_max


def get_L_crit(a, b, zs_max):
    '''
    Returns the critical separation, that is,
    the smallest L such that V(L) = 0.
    '''
    zs_UV, zs_IR = 0.001, zs_max
    with torch.no_grad():
        V_IR = integrate_V(a, b, zs_IR)
        V_UV = integrate_V(a, b, zs_UV)
        assert V_IR > 0 and V_UV < 0
        while zs_IR - zs_UV > 1e-8:
            zs_mid = (zs_UV + zs_IR) / 2
            V_mid = integrate_V(a, b, zs_mid)
            if V_mid > 0:
                zs_IR = zs_mid
            else:
                zs_UV = zs_mid
        zs_mid = (zs_UV + zs_IR) / 2
        L_crit = integrate_L(a, b, zs_mid)
    zs_mid = as_tensor(zs_mid)
    return zs_mid, L_crit


def integrate_L(a, b, zs):
    '''
    This computes the dimensionless combination T*L,
    where T = 1/(pi*z_h).
    '''
    zs = as_tensor(zs)
    y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
    z = zs * (1 - y) * (1 + y)
    sqrtg = eval_g(b, z).sqrt()
    f_over_fs = eval_f(a, z) / eval_f(a, zs)
    integrand = sqrtg / torch.sqrt(f_over_fs / ((1 - y)**4 * (1 + y)**4) - 1)
    integrand *= y
    # We extrapolate to y=0
    y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
    deriv = (integrand[1] - integrand[0]) / (y[1] - y[0])
    integrand_0 = deriv * (-y[0]) + integrand[0]
    integrand = torch.cat((integrand_0.unsqueeze(-1), integrand))
    # Add analytically known value at y=1
    y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
    integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
    # Integrate
    L = 4 * zs * torch.trapz(integrand, y) / np.pi
    assert not torch.isnan(L), f'integrate_L({zs}) = {L} for a = {a} b = {b}'
    return L


def integrate_dL(a, b, zs):
    '''
    This computes the derivative of T*L w.r.t. z_*/z_h.
    '''
    zs = as_tensor(zs)
    y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
    z = zs * (1 - y) * (1 + y)
    fs = eval_f(a, zs)
    g = eval_g(b, z)
    f_over_fs = eval_f(a, z) / fs
    df = eval_df(a, z)
    dfs = eval_df(a, zs)
    dg = eval_dg(b, z)
    integrand = (zs**4 / z**4 * f_over_fs * (zs * dfs / fs + 2 + z * dg / g)
                 - zs**4 / z**3 * df / fs - 2 - z * dg / g)
    integrand *= 2 * torch.sqrt(1 - z / zs) * torch.sqrt(g)
    integrand /= (zs**4 / z**4 * f_over_fs - 1)**1.5
    # Extrapolate to y=0
    y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
    deriv = (integrand[1] - integrand[0]) / (y[1] - y[0])
    integrand_0 = deriv * (-y[0]) + integrand[0]
    integrand = torch.cat((integrand_0.unsqueeze(-1), integrand))
    # Add known value for y=1
    y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
    integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
    # Integrate
    dL = torch.trapz(integrand, y) / np.pi
    assert not torch.isnan(dL), (
        f'integrate_dL({zs}) = {dL} for a = {a} b = {b}'
    )
    return dL


def integrate_V(a, b, zs):
    V_c = integrate_V_connected(a, b, zs)
    # Then we subtract the disconnected configuration
    V_d = integrate_V_disconnected(a, b, zs)
    return V_c - V_d


def integrate_V_connected(a, b, zs):
    '''
    This computes the connected contribution of V/T,
    where T = 1/(pi*z_h).
    '''
    zs = as_tensor(zs)
    y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
    z = zs * (1 - y) * (1 + y)
    f = eval_f(a, z)
    fg = f * eval_g(b, z)
    f_over_fs = f / eval_f(a, zs)
    integrand = torch.sqrt(fg) / ((1 - y)**2 * (1 + y)**2) * \
        (1 / torch.sqrt(1 - (1 - y)**4 * (1 + y)**4 / f_over_fs) - 1) * y
    # We extrapolate to y=0
    y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
    deriv = (integrand[1] - integrand[0]) / (y[1] - y[0])
    integrand_0 = deriv * (-y[0]) + integrand[0]
    integrand = torch.cat((integrand_0.unsqueeze(-1), integrand))
    # Add analytically known value at y=1
    y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
    integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
    # Integrate
    V = np.pi * 4 * torch.trapz(integrand, y) / zs
    assert not torch.isnan(V), (
        f'integrate_V_connected({zs}) = {V} for a = {a} b = {b}'
    )
    return V


def integrate_V_disconnected(a, b, zs):
    '''
    This computes the disconnected contribution of V/T,
    where T = 1/(pi*z_h).
    '''
    # Coordinate is y = (1 - z) / (1 - zs)
    y = torch.linspace(0.001, 1, steps=1000, dtype=dtype)
    z = 1 - (1 - zs) * y
    fg = eval_f(a, z) * eval_g(b, z)
    integrand = torch.sqrt(fg) / z**2
    # Extrapolate to y=0
    y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
    deriv = (integrand[1] - integrand[0]) / (y[1] - y[0])
    integrand_0 = deriv * (-y[0]) + integrand[0]
    integrand = torch.cat((integrand_0.unsqueeze(-1), integrand))
    V = np.pi * 2 * (1 - zs) * torch.trapz(integrand, y)
    assert not torch.isnan(V), (
        f'integrate_V_disconnected({zs}) = {V} for a = {a} b = {b}'
    )
    return V


def eval_f(a, z):
    z = as_tensor(z)
    out = torch.zeros_like(z)
    _a = torch.cat((torch.tensor([1.0]), a))
    for i, ci in enumerate(_a):
        if i == 4:
            out += -4 * ci * z**4 * torch.log(z)
        else:
            out += 4 * ci * (z**4 - z**i) / (i - 4)
    return out


def eval_df(a, z):
    z = as_tensor(z)
    out = torch.zeros_like(z)
    _a = torch.cat((torch.tensor([1.0]), a))
    for i, ci in enumerate(_a):
        out += -4 * ci * z**i
    out += 4 * eval_f(a, z)
    out /= z
    # TODO: add z->0 limit exactly
    return out


def eval_ddf(a, z):
    z = as_tensor(z)
    out = torch.zeros_like(z)
    _a = torch.cat((torch.tensor([1.0]), a))
    for i, ci in enumerate(_a[1:]):
        out += (i + 1) * ci * z**i
    out = 3 * eval_df(a, z) / z - 4 * out / z
    return out


def eval_b(b, z):
    z = as_tensor(z)
    out = torch.zeros_like(z)
    _b = torch.cat((torch.tensor([1.0]), b))
    for i, ci in enumerate(_b):
        out += ci * z**i
    return out


def eval_db(b, z):
    z = as_tensor(z)
    out = torch.zeros_like(z)
    _b = torch.cat((torch.tensor([1.0]), b))
    for i, ci in enumerate(_b[1:]):
        out += (i + 1) * ci * z**i
    return out


def eval_g(b, z):
    return eval_b(b, z) / ((1 - z) * (1 + z) * (1 + z**2))


def eval_dg(b, z):
    out = 4 * z**3 * eval_b(b, z) / ((1 - z) * (1 + z) * (1 + z**2))**2
    out += eval_db(b, z) / ((1 - z) * (1 + z) * (1 + z**2))
    return out
