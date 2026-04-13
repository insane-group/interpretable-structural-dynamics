
import torch
import torch.nn as nn
class SATMDFrequencyGroundTruth(nn.Module):
    def __init__(self, m1, m2, c1, cp, k1, kp, C_p, e_pz, R, L, force_amp=1.0):
        super().__init__()

        # parameters
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.c1 = float(c1)
        self.cp = float(cp)
        self.k1 = float(k1)
        self.kp = float(kp)
        self.C_p = float(C_p)
        self.e_pz = float(e_pz)
        self.R = float(R)
        self.L = float(L)
        self.force_amp = float(force_amp)

    def dynamic_stiffness(self, omega, device=None, dtype=torch.complex64):

        # calculate the stifness matrix Z
        if not torch.is_tensor(omega):
            omega = torch.tensor(omega, dtype=torch.float32, device=device)

        omega = omega.to(device=device)
        j = torch.tensor(1j, dtype=dtype, device=device)

        w = omega.to(dtype=torch.float32)
        w2 = (w ** 2).to(dtype=torch.float32)

        m1 = self.m1
        m2 = self.m2
        c1 = self.c1
        cp = self.cp
        k1 = self.k1
        kp = self.kp
        C_p = self.C_p
        e = self.e_pz
        R = self.R
        L = self.L

        z11 = (k1 + kp - m1 * w2) + j * w * (c1 + cp)
        z12 = (-kp) - j * w * cp
        z13 = (e / C_p) * torch.ones_like(w, dtype=dtype)

        z21 = (-kp) - j * w * cp
        z22 = (kp - m2 * w2) + j * w * cp
        z23 = (-e / C_p) * torch.ones_like(w, dtype=dtype)

        z31 = (e / C_p) * torch.ones_like(w, dtype=dtype)
        z32 = (-e / C_p) * torch.ones_like(w, dtype=dtype)
        z33 = (1.0 / C_p - L * w2) + j * w * R

        # build the stiffness matrix
        Z = torch.stack([
            torch.stack([z11, z12, z13], dim=-1),
            torch.stack([z21, z22, z23], dim=-1),
            torch.stack([z31, z32, z33], dim=-1),
        ], dim=-2)

        return Z

    def forcing_vector(self, omega, device=None, dtype=torch.complex64):

        # calculates the forcing vector (in this case it is applied to the first DOF)
        if not torch.is_tensor(omega):
            omega = torch.tensor(omega, dtype=torch.float32, device=device)
        omega = omega.to(device=device)

        F = torch.zeros((*omega.shape, 3), dtype=dtype, device=device)
        F[..., 0] = self.force_amp
        return F

    def solve_frf(self, omega, device=None, dtype=torch.complex64):

        # solve the equation Z(omega) * X(omega) = F(omega)
        Z = self.dynamic_stiffness(omega, device=device, dtype=dtype)
        F = self.forcing_vector(omega, device=device, dtype=dtype)

        X = torch.linalg.solve(Z, F.unsqueeze(-1)).squeeze(-1)
        return X  # [..., 3] complex -> [U1, U2, Q2]

    def structural_accel_frf(self, omega, dtype=torch.complex64):

        X = self.solve_frf(omega, dtype=dtype)   # [U1, U2, Q2]
        U1 = X[..., 0]
        U2 = X[..., 1]

        w2 = (omega ** 2).to(U1.dtype)

        A1 = -w2 * U1
        A2 = -w2 * U2

        return torch.stack([A1, A2], dim=-1)

    def forward(self, omega):
        return self.solve_frf(omega)
