import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from project.models.physics_informed_satmd import SATMDFrequencyRelativeParameterModel
from project.models.satmd_groundtruth import SATMDFrequencyGroundTruth
from project.models.rl_inverse_net import RLInverseNet

def train_frequency_model(model,groundtruth_model,omega_grid,epochs=5000,lr=1e-3,print_every=100,lambda_w=1e-8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    with torch.no_grad():
        Y_true = groundtruth_model.solve_frf(omega_grid)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        Y_pred = model.solve_frf(omega_grid)

        # bound the loss
        scale = Y_true.abs().max().detach().clamp_min(1e-8)
        loss_data = (mse(Y_pred.real / scale, Y_true.real / scale) +mse(Y_pred.imag /scale, Y_true.imag / scale))
        reg = torch.zeros((), device=omega_grid.device)
        for p in model.parameters():
            reg = reg + torch.sum(p**2)

        # data + regularizer
        loss = loss_data + lambda_w * reg
        loss.backward()
        optimizer.step()

        # re-calculate the correction
        m2_eff, kp_eff, R_eff, L_eff = model.effective_params()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | "
                f"Loss = {loss.item():.6e} | "
                f"R = {R_eff.item():.6f} | "
                f"L = {L_eff.item():.6e} | "
                f"m2 = {m2_eff.item():.6f} | "
                f"kp = {kp_eff.item():.6f}"
            )

    return model

def print_parameters(model=None):
    with torch.no_grad():
        m2_eff, kp_eff, R_eff, L_eff = model.effective_params()

    print("\nIdentified parameters")
    print(f"R_est  = {R_eff.item()}")
    print(f"L_est  = {L_eff.item()}")
    print(f"m2_est = {m2_eff.item()}")
    print(f"kp_est = {kp_eff.item()}")


def print_relative_error(groundtruth_model=None, relative_model=None):
    if groundtruth_model == None or relative_model == None:
        print("Need two models")
    else:

        with torch.no_grad():
            Y_true = groundtruth_model.solve_frf(omega_t)
            Y_pred = model.solve_frf(omega_t)

        rel_err = torch.linalg.norm(Y_pred - Y_true) / torch.linalg.norm(Y_true)
        print("Relative FRF error:", rel_err.item())

def plot_frf(groundtruth_model=None, relative_model=None, omega_t=None, freq_hz=None, save_path=None, dpi=300):
    with torch.no_grad():
        Y_true = groundtruth_model.solve_frf(omega_t).cpu()
        Y_pred = relative_model.solve_frf(omega_t).cpu()

    plt.figure(figsize=(10, 4))
    plt.plot(freq_hz, torch.abs(Y_true).numpy(), label="True")
    plt.plot(freq_hz, torch.abs(Y_pred).numpy(), "--", label="Pred")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FRF|")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")


def acceleration_to_features(acc):

    acc_real = acc.real.reshape(-1)
    acc_imag = acc.imag.reshape(-1)
    return torch.cat([acc_real, acc_imag], dim=0)

@torch.no_grad()
def build_inverse_acc_dataset(model_ctor,fixed_params,omega_grid,train_rl_pairs,samples_per_pair=200,R_jitter=2.0,L_jitter=10e-3,device="cpu",dtype=torch.complex64):
    omega_grid = omega_grid.to(device)

    X_list = []
    acc_list = []
    RL_list = []

    R_values = [p[0] for p in train_rl_pairs]
    L_values = [p[1] for p in train_rl_pairs]

    norm_info = {
        "R_min": max(0.0, min(R_values) - 5.0),
        "R_max": max(R_values) + 5.0,
        "L_min": max(0.0, min(L_values) - 20e-3),
        "L_max": max(L_values) + 20e-3,
    }

    for R0, L0 in train_rl_pairs:
        for _ in range(samples_per_pair):
            R = R0 + torch.empty(1).uniform_(-R_jitter, R_jitter).item()
            L = L0 + torch.empty(1).uniform_(-L_jitter, L_jitter).item()

            R = max(0.0, R)
            L = max(0.0, L)

            sys = model_ctor(**fixed_params, R=R, L=L).to(device)
            acc = sys.structural_accel_frf(omega_grid, dtype=dtype)

            feat = acceleration_to_features(acc)

            X_list.append(feat)
            acc_list.append(acc)
            RL_list.append(torch.tensor([R, L], dtype=torch.float32, device=device))

    X = torch.stack(X_list, dim=0)
    acc_all = torch.stack(acc_list, dim=0)
    RL_true = torch.stack(RL_list, dim=0)

    return X, acc_all, RL_true, norm_info

def denormalize_RL(pred_norm, norm_info):
    R = pred_norm[:, 0] * (norm_info["R_max"] - norm_info["R_min"]) + norm_info["R_min"]
    L = pred_norm[:, 1] * (norm_info["L_max"] - norm_info["L_min"]) + norm_info["L_min"]
    return R, L

def structural_accel_from_RL(omega, fixed_params, R, L, dtype=torch.complex64):

        m1   = fixed_params["m1"]
        m2   = fixed_params["m2"]
        c1   = fixed_params["c1"]
        cp   = fixed_params["cp"]
        k1   = fixed_params["k1"]
        kp   = fixed_params["kp"]
        C_p  = fixed_params["C_p"]
        e_pz = fixed_params["e_pz"]
        force_amp = fixed_params["force_amp"]

        w = omega.to(torch.float32)
        w2 = w ** 2
        j = torch.tensor(1j, dtype=dtype, device=omega.device)

        z11 = (k1 + kp - m1 * w2) + j * w * (c1 + cp)
        z12 = (-kp) - j * w * cp
        z13 = (e_pz / C_p) * torch.ones_like(w, dtype=dtype)

        z21 = (-kp) - j * w * cp
        z22 = (kp - m2 * w2) + j * w * cp
        z23 = (-e_pz / C_p) * torch.ones_like(w, dtype=dtype)

        z31 = (e_pz / C_p) * torch.ones_like(w, dtype=dtype)
        z32 = (-e_pz / C_p) * torch.ones_like(w, dtype=dtype)
        z33 = (1.0 / C_p - L * w2) + j * w * R

        Z = torch.stack([
            torch.stack([z11, z12, z13], dim=-1),
            torch.stack([z21, z22, z23], dim=-1),
            torch.stack([z31, z32, z33], dim=-1),
        ], dim=-2)

        F = torch.zeros((omega.shape[0], 3), dtype=dtype, device=omega.device)
        F[:, 0] = force_amp

        X = torch.linalg.solve(Z, F.unsqueeze(-1)).squeeze(-1)
        U1 = X[:, 0]
        U2 = X[:, 1]

        A1 = -(w2.to(U1.dtype)) * U1
        A2 = -(w2.to(U2.dtype)) * U2

        return torch.stack([A1, A2], dim=-1)

def train_inverse_via_acceleration_loss(model,X_train,acc_train,omega_grid,fixed_params,norm_info,epochs=300,lr=1e-3,print_every=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_loss = 0.0

        for i in range(X_train.shape[0]):
            x = X_train[i:i+1]
            acc_true = acc_train[i]

            pred_norm = model(x)
            R_pred, L_pred = denormalize_RL(pred_norm, norm_info)

            acc_pred = structural_accel_from_RL(omega=omega_grid,fixed_params=fixed_params,R=R_pred[0],L=L_pred[0])

            scale = acc_true.abs().max().detach().clamp_min(1e-8)

            loss_i = (mse(acc_pred.real / scale, acc_true.real / scale) + mse(acc_pred.imag / scale, acc_true.imag / scale))

            train_loss = train_loss + loss_i

        train_loss = train_loss / X_train.shape[0]
        train_loss.backward()
        optimizer.step()


        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"train_loss = {train_loss.item():.6e} | "
            )

    return model

@torch.no_grad()
def predict_RL_from_acceleration(model, acc_curve, norm_info):

    feat = acceleration_to_features(acc_curve).unsqueeze(0)
    pred_norm = model(feat).squeeze(0)

    R_pred = pred_norm[0].item() * (norm_info["R_max"] - norm_info["R_min"]) + norm_info["R_min"]
    L_pred = pred_norm[1].item() * (norm_info["L_max"] - norm_info["L_min"]) + norm_info["L_min"]

    return R_pred, L_pred

@torch.no_grad()
def evaluate_on_pairs(model,model_ctor,fixed_params,omega_grid,eval_rl_pairs,norm_info,device="cpu"):
    results = []

    model.eval()

    for R_true, L_true in eval_rl_pairs:
        sys = model_ctor(**fixed_params,R=R_true,L=L_true).to(device)

        acc_true = sys.structural_accel_frf(omega_grid)

        R_pred, L_pred = predict_RL_from_acceleration(model=model,acc_curve=acc_true,norm_info=norm_info)

        results.append({
            "R_true": R_true,
            "L_true": L_true,
            "R_pred": R_pred,
            "L_pred": L_pred,
            "R_abs_err": abs(R_pred - R_true),
            "L_abs_err": abs(L_pred - L_true),
        })

    return results


if __name__ == "__main__":

    m1 = 0.85
    m2 = 0.518
    m2_nom = 0.300

    c1 = 10.0
    cp = 4.0

    k1 = 194e3
    kp = 111e3
    kp_nom = 50e3

    e_pz = 0.65
    C_p = 27e-6

    R_true = 22.0
    L_true = 64e-3

    # choose a nearby but wrong value
    R_nom = 20.0
    L_nom = 50e-3

    f_min_hz = 40.0
    f_max_hz = 130.0
    n_freq = 60

    freq_hz = np.linspace(f_min_hz, f_max_hz, n_freq)
    omega = 2.0 * np.pi * freq_hz
    force_amp = 1.0
    dtype_real = torch.float32
    dtype_complex = torch.complex64
    device = "cpu"

    omega_t = torch.tensor(omega, dtype=dtype_real, device=device)
    freq_hz_t = torch.tensor(freq_hz, dtype=dtype_real, device=device)

    groundtruth_model = SATMDFrequencyGroundTruth(m1=m1,m2=m2,c1=c1,cp=cp,k1=k1,kp=kp,C_p=C_p,e_pz=e_pz,R=R_true,L=L_true,force_amp=force_amp).to(device)

    model = SATMDFrequencyRelativeParameterModel(m1=m1,m2_nom=m2_nom,c1=c1,cp=cp,k1=k1,kp_nom=kp_nom,C_p=C_p,e_pz=e_pz,R_nom=R_nom,L_nom=L_nom,force_amp=force_amp).to(device)

    # load the pretrained model
    model.load_state_dict(torch.load("../../../models/electrical_exp4_correction_net.pth", map_location=device, weights_only=True))
    model.eval()

    # train the model

    # model = train_frequency_model(
    #     model=model,
    #     groundtruth_model=groundtruth_model,
    #     omega_grid=omega_t,
    #     epochs=10000,
    #     lr=1e-3,
    #     print_every=100,
    # )

    # save_dir = "../../../models/"
    # os.makedirs(save_dir, exist_ok=True)

    # save_path = os.path.join(save_dir, "electrical_exp4_correction_net.pth")
    # torch.save(model.state_dict(), save_path)

    # print(f"Saved model weights to: {save_path}")

    print_parameters(model=model)
    print_relative_error(groundtruth_model=groundtruth_model, relative_model=model)

    save_path = "../../../logs/freq_response_comparison.png"
    plot_frf(groundtruth_model=groundtruth_model, relative_model=model, omega_t=omega_t, freq_hz=freq_hz, save_path=save_path)

    # these are the 30 pairs for training
    train_rl_pairs = [
        (5.0,   14e-3),
        (7.0,   26e-3),
        (11.0,  42e-3),
        (13.0,  48e-3),
        (15.0,  52e-3),
        (17.0,  64e-3),
        (19.0,  80e-3),
        (21.0,  92e-3),
        (23.0,  108e-3),
        (26.0,  116e-3),
        (27.0,  124e-3),
        (30.0,  148e-3),
        (34.0,  168e-3),
        (36.0,  184e-3),
        (40.0,  200e-3),
        (44.0,  212e-3),
        (46.0,  220e-3),
        (50.0,  236e-3),
        (54.0,  260e-3),
        (56.0,  272e-3),
        (62.0,  312e-3),
        (68.0,  336e-3),
        (70.0,  348e-3),
        (75.0,  372e-3),
        (78.0,  384e-3),
        (80.0,  392e-3),
        (88.0,  420e-3),
        (90.0,  432e-3),
        (92.0,  448e-3),
        (98.0,  480e-3),
    ]

    fixed_params = dict(m1=m1,m2=m2,c1=c1,cp=cp,k1=k1,kp=kp,C_p=C_p,e_pz=e_pz,force_amp=force_amp)

    X, acc, y, norm_info = build_inverse_acc_dataset(model_ctor=SATMDFrequencyGroundTruth,fixed_params=fixed_params,omega_grid=omega_t,train_rl_pairs=train_rl_pairs,samples_per_pair=5,R_jitter=2.0,L_jitter=10e-3,device=device)

    n = X.shape[0]
    perm = torch.randperm(n, device=X.device)

    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    X_train, acc_train, y_train = X[train_idx], acc[train_idx], y[train_idx]
    X_val, acc_val, y_val = X[val_idx], acc[val_idx], y[val_idx]

    regressor = RLInverseNet(input_dim=X.shape[1]).to(device)

    # train the model
    # regressor = train_inverse_via_acceleration_loss(model=regressor,X_train=X_train, acc_train=acc_train,omega_grid = omega_t,fixed_params=fixed_params,norm_info=norm_info,epochs=5000,lr=5e-4,print_every=20)

    # # save the model
    # save_dir = "../../../models/"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "rl_inverse_net_1_256_5000_V2.pth")
    # torch.save(regressor.state_dict(), save_path)

    # load the model
    save_dir = "../../../models/"
    save_path = os.path.join(save_dir, "rl_inverse_net_5_256_5000.pth")
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    regressor.load_state_dict(state_dict)

    eval_rl_pairs = [
        (8.0,  32e-3),
        (10.0, 50e-3),
        (22.0, 64e-3),
        (25.0, 100e-3),
        (35.0, 192e-3),
        (60.0, 300e-3),
    ]

    results = evaluate_on_pairs(
        model=regressor,
        model_ctor=SATMDFrequencyGroundTruth,
        fixed_params=fixed_params,
        omega_grid=omega_t,
        eval_rl_pairs=eval_rl_pairs,
        norm_info=norm_info,
        device=device,
    )

    for r in results:
        print(r)


