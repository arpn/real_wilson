import logging
import torch
import torch.nn as nn
import numpy as np
from numpy.random import random
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
        self.logcoef = nn.Parameter(torch.normal(0.0, std, size=(1,), dtype=dtype)[0])
        '''
        The lattice data is supposed to be shifted such that it behaves
        correctly in the UV. `self.shift` holds that parameter.
        '''
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=dtype))

    def as_tensor(self, tensor):
        '''
        Converts `tensor` to a torch.tensor if it
        is not already a tensor.
        '''
        return tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor, dtype=dtype)

    def forward(self, Ls):
        '''
        Initial version with torch.trapz
        instead of torchdiffeq.
        '''
        V = torch.zeros_like(Ls, dtype=dtype)

        zs_max, L_max = self.get_L_max()
        curve = interp1d([0.0, L_max.item()], [0.0, zs_max.item()])
        zs_crit, L_crit = self.get_L_crit(zs_max)

        for i, L in enumerate(Ls):
            if L < L_crit:
                # String is not broken
                init = curve(L.item())
                zs = self.find_zs_newton(L, init)
                assert zs > 0, f'zs is negative: {zs} for L = {L}'
                V[i] = self.integrate_V(zs)
            else:
                # String is broken
                V[i] = self.as_tensor(0.0)
            assert not torch.isnan(V[i])
        return V

    def find_zs_newton(self, L, init, max_steps=25):
        init = self.as_tensor(init)
        zs = [init]
        _L = self.integrate_L(zs[-1])
        for i in range(max_steps):
            dL = self.integrate_dL(zs[-1])
            zs.append(zs[-1] - (_L - L) / dL)
            assert 0 < zs[-1] < 1 and not torch.isnan(zs[-1]), f'Something wrong in Newton:\n\tzs = {[_zs.item() for _zs in zs]}\n\tL = {L}\n\t_L = {_L}\n\tdL = {dL}'
            _L = self.integrate_L(zs[-1])
            diff = torch.abs(_L - L)
            if diff < 1e-8:
                return zs[-1]
        # Max steps was reached with no convergence
        assert False, f'Newton\'s method failed to converge in {max_steps} iterations for L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}.'

    def get_L_max(self):
        '''
        Returns the point where L is maximal such that
        zs is still real. This is the last point on the
        real axis along the real L curve.
        '''
        zs_UV, zs_IR = 0.001, 0.999
        with torch.no_grad():
            dL_IR = self.integrate_dL(zs_IR)
            dL_UV = self.integrate_dL(zs_UV)
            assert dL_IR < 0 and dL_UV > 0
            while zs_IR - zs_UV > 1e-8:
                zs_mid = (zs_UV + zs_IR) / 2
                dL_mid = self.integrate_dL(zs_mid)
                if dL_mid < 0:
                    zs_IR = zs_mid
                else:
                    zs_UV = zs_mid
            zs_mid = (zs_UV + zs_IR) / 2
            L_max = self.integrate_L(zs_mid)
        zs_mid = torch.tensor(zs_mid, dtype=dtype)
        return zs_mid, L_max

    def get_L_crit(self, zs_max):
        '''
        Returns the critical separation, that is,
        the smallest L such that V(L) = 0.
        '''
        zs_UV, zs_IR = 0.001, zs_max
        with torch.no_grad():
            V_IR = self.integrate_V(zs_IR)
            V_UV = self.integrate_V(zs_UV)
            assert V_IR > 0 and V_UV < 0
            while zs_IR - zs_UV > 1e-8:
                zs_mid = (zs_UV + zs_IR) / 2
                V_mid = self.integrate_V(zs_mid)
                if V_mid > 0:
                    zs_IR = zs_mid
                else:
                    zs_UV = zs_mid
            zs_mid = (zs_UV + zs_IR) / 2
            L_crit = self.integrate_L(zs_mid)
        zs_mid = self.as_tensor(zs_mid)
        return zs_mid, L_crit

    def integrate_L(self, zs):
        '''
        This computes the dimensionless combination T*L,
        where T = 1/(pi*z_h).
        '''
        zs = self.as_tensor(zs)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
        z = zs * (1 - y) * (1 + y)
        sqrtg = self.eval_g(z).sqrt()
        f_over_fs = self.eval_f(z) / self.eval_f(zs)
        integrand = sqrtg / \
            torch.sqrt(f_over_fs / ((1 - y)**4 * (1 + y)**4) - 1) * y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
        # Integrate
        L = 4 * zs * torch.trapz(integrand, y) / np.pi
        assert not torch.isnan(L), f'integrate_L({zs}) = {L} for a = {self.a} b = {self.b}'
        return L

    def integrate_dL(self, zs):
        '''
        This computes the derivative of T*L w.r.t. z_*/z_h.
        '''
        zs = self.as_tensor(zs)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
        z = zs * (1 - y) * (1 + y)
        fs = self.eval_f(zs)
        g = self.eval_g(z)
        f_over_fs = self.eval_f(z) / fs
        df = self.eval_df(z)
        dfs = self.eval_df(zs)
        dg = self.eval_dg(z)
        integrand = (zs**4 / z**4 * f_over_fs * (zs * dfs / fs + 2 + z * dg / g) - zs**4 / z**3 * df / fs - 2 - z * dg / g)
        integrand *= 2 * torch.sqrt(1 - z / zs) * torch.sqrt(g) / (zs**4 / z**4 * f_over_fs - 1)**1.5
        # Extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add known value for y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
        # Integrate
        dL = torch.trapz(integrand, y) / np.pi
        assert not torch.isnan(dL), f'integrate_dL({zs}) = {dL} for a = {self.a} b = {self.b}'
        self.dL_int = integrand
        return dL

    def integrate_V(self, zs):
        V_c = self.integrate_V_connected(zs)
        # Then we subtract the disconnected configuration
        V_d = self.integrate_V_disconnected(zs)
        return V_c - V_d

    def integrate_V_connected(self, zs):
        '''
        This computes the connected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
        zs = self.as_tensor(zs)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
        z = zs * (1 - y) * (1 + y)
        f = self.eval_f(z)
        fg = f * self.eval_g(z)
        f_over_fs = f / self.eval_f(zs)
        integrand = torch.sqrt(fg) / ((1 - y)**2 * (1 + y)**2) * \
            (1 / torch.sqrt(1 - (1 - y)**4 * (1 + y)**4 / f_over_fs) - 1) * y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
        # Integrate
        coef = self.logcoef.exp()
        V = coef * np.pi * 4 * torch.trapz(integrand, y) / zs
        assert not torch.isnan(V), f'integrate_V_connected({zs}) = {V} for a = {self.a} b = {self.b}'
        self.Vc_int = integrand
        return V

    def integrate_V_disconnected(self, zs):
        '''
        This computes the disconnected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
        # Coordinate is y = (1 - z) / (1 - zs)
        y = torch.linspace(0.001, 1, steps=1000, dtype=dtype)
        z = 1 - (1 - zs) * y
        fg = self.eval_f(z) * self.eval_g(z)
        integrand = torch.sqrt(fg) / z**2
        # NOTE: This assumes that f(z)*g(z)->1 when z->1
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat((torch.tensor([1.0], dtype=dtype), integrand))
        coef = self.logcoef.exp()
        V = coef * np.pi * 2 * (1 - zs) * torch.trapz(integrand, y)
        assert not torch.isnan(V), f'integrate_V_disconnected({zs}) = {V} for a = {self.a} b = {self.b}'
        self.Vd_int = integrand
        return V

    def eval_f(self, z):
        z = self.as_tensor(z)
        out = torch.zeros_like(z)
        _a = torch.cat((torch.tensor([1.0], dtype=dtype), self.a))
        for i, ci in enumerate(_a):
            for j, cj in enumerate(_a):
                if i + j == 4:
                    out += -4 * ci * cj * z**4 * torch.log(z)
                else:
                    out += 4 * ci * cj * (z**4 - z**(i + j)) / (i + j - 4)
        return out

    def eval_df(self, z):
        z = self.as_tensor(z)
        out = torch.zeros_like(z)
        _a = torch.cat((torch.tensor([1.0], dtype=dtype), self.a))
        for i, ci in enumerate(_a):
            for j, cj in enumerate(_a):
                out -= 4 * ci * cj * z**(i + j)
        out += 4 * self.eval_f(z)
        out /= z
        # TODO: add z->0 limit exactly
        # This limit is to linear order:
        # df = 8*a[0]/3 + (4*a[0]**2+8*a[1])*z
        return out

    def eval_b(self, z):
        z = self.as_tensor(z)
        out = torch.zeros_like(z)
        _b = torch.cat((torch.tensor([1.0], dtype=dtype),
                        self.b,
                        self.a.sum().unsqueeze(-1) - self.b.sum()))
        for i, ci in enumerate(_b):
            for j, cj in enumerate(_b):
                out += ci * cj * z**(i + j)
        return out

    def eval_db(self, z):
        z = self.as_tensor(z)
        x = torch.zeros_like(z)
        dx = torch.zeros_like(z)
        _b = torch.cat((torch.tensor([1.0], dtype=dtype),
                        self.b,
                        self.a.sum().unsqueeze(-1) - self.b.sum()))
        for i, ci in enumerate(_b):
            x += ci * z**i
        for i, ci in enumerate(_b[1:]):
            dx += (i + 1) * ci * z**i
        out = 2 * dx * x
        return out

    def eval_g(self, z):
        return self.eval_b(z) / ((1 - z) * (1 + z) * (1 + z**2))

    def eval_dg(self, z):
        out = 4 * z**3 * self.eval_b(z) / ((1 - z) * (1 + z) * (1 + z**2))**2
        out += self.eval_db(z) / ((1 - z) * (1 + z) * (1 + z**2))
        return out
