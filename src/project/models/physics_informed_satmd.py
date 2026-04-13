import torch
import torch.nn as nn

class SATMDFrequencyRelativeParameterModel(nn.Module):
    def __init__(self,m1, m2_nom, c1, cp, k1, kp_nom, C_p, e_pz,R_nom, L_nom, force_amp=1.0):
        super().__init__()

        # parameters
        self.m1 = float(m1)
        self.c1 = float(c1)
        self.cp = float(cp)
        self.k1 = float(k1)
        self.C_p = float(C_p)
        self.e_pz = float(e_pz)
        self.force_amp = float(force_amp)

        # known parameters we want to optimize
        self.m2_nom = float(m2_nom)
        self.kp_nom = float(kp_nom)
        self.R_nom  = float(R_nom)
        self.L_nom  = float(L_nom)

        # correction of the parameters
        self.corr_m2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.corr_kp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.corr_R  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.corr_L  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def effective_params(self):
        # add the correction to the initial value
        m2 = self.m2_nom * (1.0 + self.corr_m2)
        kp = self.kp_nom * (1.0 + self.corr_kp)
        R  = self.R_nom  * (1.0 + self.corr_R)
        L  = self.L_nom  * (1.0 + self.corr_L)
        return m2, kp, R, L

    def dynamic_stiffness(self, omega, dtype=torch.complex64):
        w = omega.to(torch.float32)
        w2 = w ** 2
        j = torch.tensor(1j, dtype=dtype, device=omega.device)

        m2, kp, R, L = self.effective_params()

        z11 = (self.k1 + kp - self.m1 * w2) + j * w * (self.c1 + self.cp)
        z12 = (-kp) - j * w * self.cp
        z13 = (self.e_pz / self.C_p) * torch.ones_like(w, dtype=dtype)

        z21 = (-kp) - j * w * self.cp
        z22 = (kp - m2 * w2) + j * w * self.cp
        z23 = (-self.e_pz / self.C_p) * torch.ones_like(w, dtype=dtype)

        z31 = (self.e_pz / self.C_p) * torch.ones_like(w, dtype=dtype)
        z32 = (-self.e_pz / self.C_p) * torch.ones_like(w, dtype=dtype)
        z33 = (1.0 / self.C_p - L * w2) + j * w * R

        Z = torch.stack([
            torch.stack([z11, z12, z13], dim=-1),
            torch.stack([z21, z22, z23], dim=-1),
            torch.stack([z31, z32, z33], dim=-1),
        ], dim=-2)

        return Z

    def forcing_vector(self, omega, dtype=torch.complex64):
        F = torch.zeros((*omega.shape, 3), dtype=dtype, device=omega.device)
        F[..., 0] = self.force_amp
        return F

    def solve_frf(self, omega, dtype=torch.complex64):

        # solve the equation Z(omega) * X(omega) = F(omega)
        Z = self.dynamic_stiffness(omega, dtype=dtype)
        F = self.forcing_vector(omega, dtype=dtype)
        X = torch.linalg.solve(Z, F.unsqueeze(-1)).squeeze(-1)
        return X # [..., 3] complex -> [U1, U2, Q2]

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
