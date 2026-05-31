import numpy as np
import juliacall
import torch
from torchdiffeq import odeint
from pysr import PySRRegressor

from project.models.excitation import UFunFromSamples
from project.models.pinode_linear_3dof import PINODEFuncLinear3DOF
from project.models.pinode_nsd_3dof import PINODEFuncNSD_3DOF
from project.models.truth_nsd_3dof import TruthPhaseNSD_3DOF
from project.models.nsd_net import NSD_Net
from project.experiments.download_data import ensure_all_earthquakes


def model_accel(model, t, x, v):
    h = torch.cat([x, v], dim=-1)
    dh = model(t, h)
    ndof = x.shape[-1]
    return dh[..., ndof:]

def rollout_central_difference(model, h0, t_grid):
    dt = (t_grid[1] - t_grid[0]).to(h0)
    T = t_grid.numel()

    if h0.ndim == 1:
        h0 = h0.unsqueeze(0)
        batch = False
    else:
        batch = True

    B, D = h0.shape
    ndof = D // 2

    x0 = h0[:, :ndof]
    v0 = h0[:, ndof:]

    x_prev = x0 - dt * v0

    traj = torch.zeros((T, B, D), device=h0.device, dtype=h0.dtype)
    traj[0, :, :ndof] = x0
    traj[0, :, ndof:] = v0

    x = x0
    for n in range(T - 1):
        t = t_grid[n]
        v = (x - x_prev) / dt
        a = model_accel(model, t, x, v)

        x_next = 2 * x - x_prev + (dt ** 2) * a
        v_next = (x_next - x) / dt

        traj[n + 1, :, :ndof] = x_next
        traj[n + 1, :, ndof:] = v_next

        x_prev, x = x, x_next

    return traj if batch else traj[:, 0, :]

def nsd_force_bilinear(x1, d0=0.15, k_post=-10.0):
    excess = torch.relu(torch.abs(x1) - d0)
    return k_post * excess * torch.sign(x1)

def read_at2(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    parts = lines[3].split()
    npts = int(parts[0])
    dt_file = float(parts[1])

    data_str = " ".join(lines[4:])
    acc = np.fromstring(data_str, sep=" ")
    acc = acc[:npts]
    return acc.astype(np.float32), dt_file, npts

import numpy as np
import torch


def load_elcentro(path: str,device,dt_sim: float = 1.0 / 256.0):

    data = np.loadtxt(path)

    # El Centro file-specific sampling step if only acceleration is provided
    dt_ec = 0.02

    if data.ndim == 1:
        acc_ec = data.astype(np.float32)
        t_ec = (np.arange(len(acc_ec), dtype=np.float32) * dt_ec)
    else:
        t_ec = data[:, 0].astype(np.float32)
        acc_ec = data[:, 1].astype(np.float32)

    T_end = float(t_ec[-1])

    dt = 1.0 / 256.0
    t_full = torch.arange(0.0, T_end + dt, dt, device=device)

    u_full_np = np.interp(
        t_full.detach().cpu().numpy(),
        t_ec,
        acc_ec
    ).astype(np.float32)
    u_full = torch.tensor(u_full_np, device=device)
    return t_full, u_full



def load_kobe(path: str,device,dt_sim: float = 1.0 / 256.0):

    u_file_np, dt_file, npts = read_at2(path)

    T_end = (npts - 1) * dt_file
    t_file = np.arange(npts, dtype=np.float32) * dt_file

    t_full = torch.arange(0.0, T_end + dt_sim, dt_sim, device=device)

    u_full_np = np.interp(
        t_full.detach().cpu().numpy(),
        t_file,
        u_file_np
    ).astype(np.float32)

    u_full = torch.tensor(u_full_np, device=device)
    return t_full, u_full


if __name__ == "__main__":

    device = "cpu"

    # Parameters
    c1 = c2 = c3 = 0.015
    m1 = m2 = 9.0
    m3 = 9.2

    # Stiffness matrix K for a 3-DOF
    K = torch.tensor([
        [20.26,   -11.16,   0.0],
        [-11.16,  20.33,    -9.17],
        [0.0,      -9.17,     9.17],
    ], dtype=torch.float32, device=device)

    # Damping matrix (diagonal with c1..c3)
    C = torch.diag(torch.tensor([c1, c2, c3], dtype=torch.float32, device=device))

    # Mass Matrix
    M = (1.0/386) * torch.diag(torch.tensor([m1, m2, m3], dtype=torch.float32, device=device))

    r = torch.ones(3, device=device)
    B = -(M @ r)  # (3,)

    # Start from nominal
    M_true = M.clone()
    K_true = K.clone()
    C_true = C.clone()

    # Add controlled mismatch
    K_true = 0.97 * K_true  # stiffness is 3% lower than nominal
    C_true = 1.25 * C_true  # damping is 25% higher than nominal

    # Initial conditions (6-dim states for 3DOF)
    h0_4 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    elcentro_path, kobe_path = ensure_all_earthquakes()

    dt=1.0/256.0
    t_full, u_full= load_elcentro(elcentro_path, device, dt)
    # t_full, u_full= load_kobe(kobe_path, device, dt)

    T_train_sec = 15.0
    N_train = int(T_train_sec / dt) + 1
    t_train = t_full[:N_train]
    u_train = u_full[:N_train]


    t_test, u_test = t_train, u_train
    amp_test=150

    # Load models and run truth trajectory
    model = PINODEFuncLinear3DOF(M, K, C, B)
    true_trajectory = TruthPhaseNSD_3DOF(M_true, K_true, C_true, nsd_force_bilinear)

    true_trajectory.u_fun = UFunFromSamples(t_test, u_test)
    true_trajectory.amp =amp_test


    save_path = "../../../models/"

    save_model_path_nn1 = save_path + "linear_with_forced_vibration_base_model_15sec.pth"
    state = torch.load(save_model_path_nn1, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.amp = amp_test
    model.u_fun = UFunFromSamples(t_test, u_test)

    nn2 = NSD_Net().to(device)
    model_nsd = PINODEFuncNSD_3DOF(M, K, C, model.mlp, nn2).to(device)


    save_model_path_nn2 = save_path + "nsd_forced_vibration_base_15sec_model.pth"
    state = torch.load(save_model_path_nn2, map_location=device, weights_only=True)
    model_nsd.load_state_dict(state, strict=False)
    for p in model_nsd.mlp.parameters():
        p.requires_grad = False
    model_nsd.u_fun = UFunFromSamples(t_test, u_test)
    model_nsd.amp = amp_test


    with torch.no_grad():
        traj = odeint(true_trajectory, h0_4, t_test, method="rk4")  # (T,6)


    # Build dataset: h = truth states, y = NN2 force on DOF1
    if isinstance(traj, (list, tuple)):
        trajs = list(traj)
    else:
        trajs = [traj]

    y_targets = []

    with torch.no_grad():
        for h in trajs:
            h = h.to(device)
            if h.ndim != 2 or h.shape[1] != 6:
                raise ValueError(f"Expected (T,6), got {tuple(h.shape)}")

            x1 = h[:, 0]

            a_raw = model_nsd.nsd(h).squeeze(-1)  # (T,)
            a_vec = torch.zeros((h.shape[0], 3), device=device, dtype=h.dtype)
            a_vec[:, 0] = a_raw

            f_vec = (M @ a_vec.unsqueeze(-1)).squeeze(-1)  # (T,3)
            f1 = f_vec[:, 0]  # (T,)q

            y_targets.append(f1.detach().cpu().numpy())


    # Build dataset
    H_all = traj.detach().cpu().numpy() if torch.is_tensor(traj) else np.asarray(traj)

    if isinstance(y_targets, (list, tuple)):
        y_all = np.concatenate(y_targets, axis=0)
    else:
        y_all = y_targets.detach().cpu().numpy() if torch.is_tensor(y_targets) else np.asarray(y_targets)

    y_all = y_all.reshape(-1)

    assert H_all.ndim == 2 and H_all.shape[1] == 6, f"traj must be (N,6), got {H_all.shape}"
    assert y_all.ndim == 1 and y_all.shape[0] == H_all.shape[0], f"y must be (N,), got {y_all.shape}"

    print("Dataset:", H_all.shape, y_all.shape)
    print("y stats: mean=%.6g std=%.6g min=%.6g max=%.6g" % (
        float(y_all.mean()), float(y_all.std()), float(y_all.min()), float(y_all.max())
    ))


    # truth states from trajectory
    H_all = traj.detach().cpu().numpy() if torch.is_tensor(traj) else np.asarray(traj)
    x1_all = H_all[:, 0]

    # true NSD force from the analytical equation
    y_true_eq = -10.0 * np.maximum(np.abs(x1_all) - 0.15, 0.0) * np.sign(x1_all)

    # learned NN force targets
    y_all_pr = y_targets.detach().cpu().numpy() if torch.is_tensor(y_targets) else np.asarray(y_targets)
    y_force = np.concatenate(y_all_pr, axis=0)

    mse_true_eq = np.mean((y_true_eq - y_force)**2)
    rmse_true_eq = np.sqrt(mse_true_eq)

    print("MSE (true equation vs NN-extracted force):", mse_true_eq)
    print("RMSE (true equation vs NN-extracted force):", rmse_true_eq)

    # ALL_NAMES = ["x1", "x2", "x3", "v1", "v2", "v3"]

    # Features
    # feat_idx = [0,1,2,3,4,5]
    # feature_names = [ALL_NAMES[j] for j in feat_idx]

    # X = H_all[:, feat_idx].astype(np.float64)


    x1, x2, x3 = H_all[:, 0], H_all[:, 1], H_all[:, 2]
    v1, v2, v3 = H_all[:, 3], H_all[:, 4], H_all[:, 5]

    # drifts
    d1 = x1
    d2 = x2 - x1
    d3 = x3 - x2

    # drift velocities
    w1 = v1
    w2 = v2 - v1
    w3 = v3 - v2

    ALL_NAMES = ["d1", "d2", "d3", "w1", "w2", "w3"]

    # Features
    feat_idx = [0, 1, 2, 3, 4, 5]  # all drifts + drift-vels
    feature_names = [ALL_NAMES[j] for j in feat_idx]

    H_feat = np.column_stack([d1, d2, d3, w1, w2, w3]).astype(np.float64)
    X = H_feat[:, feat_idx]
    y = y_all.astype(np.float64)

    # Train on all data
    X_tr, y_tr = X, y

    # PySR: loss-first settings
    BASE_KW = dict(
        niterations=200,
        populations=80,
        population_size=350,
        maxsize=10,
        maxdepth=7,
        parsimony=0.0,
        binary_operators=["+", "-", "*"],
        unary_operators=["abs", "sign", "relu"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        parallelism="multithreading",
        deterministic=False,
        verbosity=1,
        random_state=42
    )

    model = PySRRegressor(**BASE_KW)
    model.fit(X_tr, y_tr)

    hof = model.get_hof()
    row = hof.loc[hof["loss"].idxmin()]

    loss = float(row["loss"])
    eq = row["equation"]

    # Map x0,x1,... to names
    eq_named = eq
    for i in reversed(range(len(feature_names))):
        eq_named = eq_named.replace(f"x{i}", feature_names[i])

    print("\nEquation (named):")
    print(eq_named)

    # Evaluate training MSE/RMSE with the best model
    yhat = model.predict(X)
    mse_raw = float(np.mean((yhat - y) ** 2))
    rmse_raw = float(np.sqrt(mse_raw))
    print("\nTraining fit:")
    print("MSE:", mse_raw)
    print("RMSE:", rmse_raw)
