import math
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import os
import matplotlib.pyplot as plt
import pysindy as ps

# Import models
from project.models.pinode_forced_4dof import PINODEFuncForcedVibration
from project.models.node_no_physics import NODEFuncNoPhysics

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

device = "cpu"
print("Using device:", device)

# Forcing: white-noise-like excitation u(t)

torch.manual_seed(0) # this is for reproduction

num_freqs = 50
freqs = torch.linspace(0.5, 20.0, num_freqs, device=device)
phases = 2.0 * math.pi * torch.rand(num_freqs, device=device)
coeffs = torch.randn(num_freqs, device=device) / math.sqrt(num_freqs)

def u_base_fun(t):

    t = t.to(device)
    # Shape (..., 1) for broadcasting
    t_exp = t.unsqueeze(-1)
    arg = 2.0 * math.pi * freqs * t_exp + phases  # shape (..., num_freqs)
    # sum over frequencies
    return (coeffs * torch.sin(arg)).sum(dim=-1)

torch.manual_seed(0)

num_freqs_test = 70
freqs_test  = torch.linspace(0.5, 30.0, num_freqs_test, device=device)
phases_test = 2.0 * math.pi * torch.rand(num_freqs_test, device=device)
coeffs_test = torch.randn(num_freqs_test, device=device) / math.sqrt(num_freqs_test)

def u_fun_cos(t):
    t = t.to(device)
    t_exp = t.unsqueeze(-1)
    arg = 2.0 * math.pi * freqs_test * t_exp + phases_test
    return (coeffs_test * torch.cos(arg)).sum(dim=-1)

# build a continuous time forcing function
def make_timeseries_fun(t_vec, u_vec):
    dt = (t_vec[1] - t_vec[0]).item() # dt
    t0 = t_vec[0].item() # first time recording
    t_end = t_vec[-1].item() # last time recording
    N  = t_vec.numel() # number of time samples

    def u_fun(t): # this function will be called by odeint during integration.
        if t > t_end: # check
            return torch.tensor(0.0, device=t.device, dtype=u_vec.dtype)
        idx = ((t - t0) / dt).long() # maps to nearest sample
        idx = torch.clamp(idx, 0, N - 1) # force time to be inside the bounds
        return u_vec[idx]
    return u_fun

def make_ground_truth_rhs(amp, u_fun):


    def rhs(t, h):
        x = h[:4]
        v = h[4:]

        # Linear restoring + damping
        lin_force = -K @ x - C @ v

        # Cubic nonlinearity on DOF 1 only
        f_nl = torch.zeros_like(x)
        f_nl[0] = -k_nl_true * x[0]**3

        # External forcing
        u_t = amp * u_fun(t)  # scalar
        f_u = B_force * u_t      # (4,)

        a = lin_force + f_nl + f_u  # since M = I

        dh = torch.zeros_like(h)
        dh[:4] = v
        dh[4:] = a
        return dh

    return rhs

@torch.no_grad()
def generate_trajectory(h0, t_grid, amp, u_fun):

    # make the groundtruth
    rhs = make_ground_truth_rhs(amp, u_fun)

    # simulate the trajectory
    sol = odeint(rhs, h0, t_grid, method='rk4')
    return sol
@torch.no_grad()
def generate_trajectory_lists(h0_list, t_train, t_full, amplitudes, u_fun):
    # for all ICs and amplitudes, create the appropriate trajectory
    for h0 in h0_list:
        for a in amplitudes:
            traj = generate_trajectory(h0, t_train, a, u_fun)
            assert traj.ndim == 2
            assert traj.shape[0] == len(t_train)
            assert traj.shape[1] == 8
            traj_list.append(traj)


    # for all ICs and amplitutes, create the appropriate full trajectory
    for h0 in h0_list:
        for a in amplitudes:
            traj = generate_trajectory(h0, t_full, a, u_fun)
            assert traj.ndim == 2
            assert traj.shape[0] == len(t_full)
            assert traj.shape[1] == 8
            traj_list_full.append(traj)

    return traj_list, traj_list_full


# Discrepancy reference (with forcing)
def discrepancy_reference(x, v, t, scheme, K, C, k_nl, B, amp, u_fun):
    # Ensure x, v are 2D: (T, ndof) even if T=1
    if x.ndim == 1:
        x = x.unsqueeze(0)   # (1, ndof)
    if v.ndim == 1:
        v = v.unsqueeze(0)

    # nonlinear force only on DOF1
    f_nl = torch.zeros_like(x)
    f_nl[:, 0] = k_nl * x[:, 0]**3

    # linear part Kx + Cv (T,4)
    lin = torch.matmul(x, K.T) + torch.matmul(v, C.T)

    # external forcing
    u_vals = amp * u_fun(t)          # (T,)
    f_u = u_vals.unsqueeze(-1) * B.unsqueeze(0)  # (T,4)

    # true acceleration
    a_true = -(lin + f_nl) + f_u

    # scheme 1 and 2 are only for example purposes
    if scheme == 1:
        a_phy = f_u
    elif scheme == 2:
        a_phy = -0.7 * lin + f_u
    elif scheme == 3: # the only case that gets used
        a_phy = -lin + f_u
    else:
        raise ValueError("scheme must be 1, 2, or 3")

    a_disc = a_true - a_phy
    return a_disc

def true_discrepancy_on_traj(scheme, traj, t_grid, amp, u_fun):
    assert traj.ndim == 2, f"traj.ndim={traj.ndim}, expected 2"
    T, D = traj.shape
    assert D == 8, f"traj.shape={traj.shape}, expected (T,8)"
    assert t_grid.shape[0] == T, f"len(t_grid)={len(t_grid)}, T={T}"

    x = traj[:, :4]   # (T,4)
    v = traj[:, 4:]   # (T,4)

    disc = discrepancy_reference(x, v, t_grid,scheme=scheme, K=K, C=C,k_nl=k_nl_true, B=B_force, amp=amp, u_fun=u_fun)

    assert disc.shape == x.shape, (f"discrepancy_reference returned {disc.shape}, expected {x.shape}")
    return disc

def train_scheme_with_disc_multiIC_multiamp_sequential(scheme, model, h0_list, amplitudes, traj_list, traj_list_full, t_train, u_fun, num_epochs=4000, lr=1e-3, log_interval=20, loss_log_path=None, loss_history_init=None, loss_log_path_20=None, loss_history_20_init=None, save_model_path=None):
    # transfer the model to the device
    model = model.to(device)

    # assign the forcing of the training
    model.u_fun = u_fun
    model.train()

    # optimizer: ADAM instead of LBFGS
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # loss function
    mse = nn.MSELoss()

    n_ics = len(h0_list)
    n_amps = len(amplitudes)

    assert len(traj_list) == n_ics * n_amps

    total_pairs = n_ics * n_amps
    epochs_per_pair = num_epochs // total_pairs
    leftover = num_epochs - epochs_per_pair * total_pairs

    # RESUME LOGIC
    if loss_history_init is None:
        loss_history_list = []
        global_epoch = 0
    else:
        if isinstance(loss_history_init, np.ndarray):
            loss_history_list = loss_history_init.tolist()
        else:
            loss_history_list = list(loss_history_init)

        if len(loss_history_list) > 0:
            global_epoch = int(loss_history_list[-1][0])
        else:
            global_epoch = 0

    if loss_history_20_init is None:
        loss_history_20_list = []
    else:
        if isinstance(loss_history_20_init, np.ndarray):
            loss_history_20_list = loss_history_20_init.tolist()
        else:
            loss_history_20_list = list(loss_history_20_init)

    print(f"Training on {n_ics} ICs × {n_amps} amplitudes = {total_pairs} pairs")
    print(f"{epochs_per_pair} epochs per pair, leftover {leftover} epoch(s)")

    pair_index = 0

    # currently only one IC used
    i_ic = 0
    h0 = h0_list[0].to(device)

    for j_amp, a in enumerate(amplitudes):
        flat_idx = i_ic * n_amps + j_amp
        traj_true = traj_list[flat_idx].to(device)
        traj_true_20 = traj_list_full[flat_idx].to(device)

        assert traj_true.ndim == 2, f"traj_true[{flat_idx}].ndim={traj_true.ndim}"
        assert traj_true.shape[0] == len(t_train), (
            f"traj_true[{flat_idx}].shape={traj_true.shape}, len(t_train)={len(t_train)}"
        )
        assert traj_true.shape[1] == 8, (
            f"traj_true[{flat_idx}].shape={traj_true.shape}, expected 8 state dims"
        )

        pair_index += 1
        this_epochs = epochs_per_pair + (leftover if pair_index == total_pairs else 0)

        # extra epochs for the last pair
        if pair_index == total_pairs:
            this_epochs = this_epochs + 1000

        print(
            f"\n=== Training on IC {i_ic}, amplitude {a} "
            f"for {this_epochs} epochs (scheme {scheme}) ==="
        )

        model.amp = float(a)

        # ---- loss normalization ----
        state_dim = traj_true.shape[1]
        ndof = state_dim // 2

        x_true = traj_true[:, :ndof]
        v_true = traj_true[:, ndof:]

        eps = 1e-8
        sx = x_true.std(dim=0, unbiased=False) + eps
        sv = v_true.std(dim=0, unbiased=False) + eps

        for local_epoch in range(1, this_epochs + 1):
            global_epoch += 1

            opt.zero_grad()

            pred = odeint(model, h0, t_train, method="rk4")  # (T, 8)

            x_pred = pred[:, :ndof]
            v_pred = pred[:, ndof:]

            loss_x = torch.mean(((x_pred - x_true) / sx) ** 2)
            loss_v = torch.mean(((v_pred - v_true) / sv) ** 2)

            loss = 1.5 * loss_x + 1.0 * loss_v
            loss.backward()
            opt.step()

            # record loss every log_interval epochs
            if global_epoch % log_interval == 0:
                with torch.no_grad():
                    t_test_end = 20.0
                    t_test = torch.linspace(
                        0.0, t_test_end, int(t_test_end / dt) + 1, device=device
                    )
                    pred_20 = odeint(model, h0, t_test, method="rk4")
                    loss_20 = mse(pred_20, traj_true_20)

                loss_history_list.append([global_epoch, float(loss.item())])
                loss_history_20_list.append([global_epoch, float(loss_20.item())])

            if local_epoch == 1 or global_epoch % 100 == 0:
                print(
                    f"[S{scheme}] Global {global_epoch:4d}, "
                    f"IC={i_ic}, amp={a}, "
                    f"total={loss.item():.4e}"
                )

        # after finishing training for amplitude 'a'
        if save_model_path is not None:
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved model to {save_model_path}")

        if loss_log_path is not None:
            np.save(loss_log_path, np.array(loss_history_list, dtype=np.float32))
            print(f"Saved partial loss history to {loss_log_path}")

        if loss_log_path_20 is not None:
            np.save(loss_log_path_20, np.array(loss_history_20_list, dtype=np.float32))
            print(f"Saved partial loss history to {loss_log_path_20}")

    # final save
    loss_history_arr = np.array(loss_history_list, dtype=np.float32)
    loss_history_20_arr = np.array(loss_history_20_list, dtype=np.float32)

    if loss_log_path is not None:
        np.save(loss_log_path, loss_history_arr)
        print(f"\nSaved loss history to: {loss_log_path}")

    if loss_log_path_20 is not None:
        np.save(loss_log_path_20, loss_history_20_arr)
        print(f"Saved loss history to {loss_log_path_20}")

    return model, loss_history_arr, loss_history_20_arr



@torch.no_grad()
def evaluate_scheme_on_amp(scheme_name, model, amp_eval, traj_test_true):
    loss_fn = nn.MSELoss()
    model.amp = float(amp_eval)

    # Predict
    pred = odeint(model, h0_2, t_test, method='rk4')

    mse_full = loss_fn(pred, traj_test_true).item()

    # split at 10s
    mid_idx = len(t_train)  # since t_train ends at 10s with same dt
    mse_0_4  = loss_fn(pred[:mid_idx], traj_test_true[:mid_idx]).item()
    mse_4_8  = loss_fn(pred[mid_idx:], traj_test_true[mid_idx:]).item()

    print(f"\n=== Evaluation for {scheme_name} (amp = {amp_eval}) ===")
    print(f"MSE 0–20s  : {mse_full:.4e}")
    print(f"MSE 0–10s  : {mse_0_4:.4e}")
    print(f"MSE 10–20s  : {mse_4_8:.4e}")


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def plot_x1_v1_a1(t, gt, pred, dt=0.01, save_path=None):

    # Extract x1 and v1
    x1_gt = gt[:, 0].detach().cpu()
    v1_gt = gt[:, 4].detach().cpu()

    x1_pr = pred[:, 0].detach().cpu()
    v1_pr = pred[:, 4].detach().cpu()

    # Numerical acceleration a1 via forward diff
    a1_gt = torch.diff(v1_gt) / dt
    a1_gt = torch.cat((a1_gt[:1], a1_gt), dim=0)  # pad to same length

    a1_pr = torch.diff(v1_pr) / dt
    a1_pr = torch.cat((a1_pr[:1], a1_pr), dim=0)

    # Plot
    plt.figure(figsize=(12, 9))

    # x1(t)
    plt.subplot(3, 1, 1)
    plt.plot(t.cpu(), x1_gt, 'k', lw=2, label="GT x1")
    plt.plot(t.cpu(), x1_pr, 'g--', lw=1.5, label="Pred x1")
    plt.ylabel("x1(t)")
    plt.grid(True)
    plt.legend()

    # v1(t)
    plt.subplot(3, 1, 2)
    plt.plot(t.cpu(), v1_gt, 'k', lw=2, label="GT v1")
    plt.plot(t.cpu(), v1_pr, 'g--', lw=1.5, label="Pred v1")
    plt.ylabel("v1(t)")
    plt.grid(True)
    plt.legend()

    # a1(t)
    plt.subplot(3, 1, 3)
    plt.plot(t.cpu(), a1_gt, 'k', lw=2, label="GT a1")
    plt.plot(t.cpu(), a1_pr, 'g--', lw=1.5, label="Pred a1")
    plt.ylabel("a1(t)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=250)

    # plt.show()

def compute_a1_from_rhs(t, gt, pred1, pred2, pred3, rhs_gt, model1, model2, model3):

    a1_gt_list = []
    a1_pred1_list = []
    a1_pred2_list = []
    a1_pred3_list = []

    with torch.no_grad():
        for k, t_k in enumerate(t):
            dh_gt = rhs_gt(t_k, gt[k])
            a1_gt_list.append(dh_gt[4].item())

            dh_pred1 = model1(t_k, pred1[k])
            a1_pred1_list.append(dh_pred1[4].item())

            dh_pred2 = model2(t_k, pred2[k])
            a1_pred2_list.append(dh_pred2[4].item())

            dh_pred3 = model3(t_k, pred3[k])
            a1_pred3_list.append(dh_pred3[4].item())

    a1_gt = np.array(a1_gt_list)
    a1_pred1 = np.array(a1_pred1_list)
    a1_pred2 = np.array(a1_pred2_list)
    a1_pred3 = np.array(a1_pred3_list)

    return a1_gt, a1_pred1, a1_pred2, a1_pred3

def plot_x1_v1_a1_three_preds(t, h0, amp, u_fun, model_1, model_2, model_3, dt=0.01, label1="Prediction 1", label2="Prediction 2", label3="Prediction 3", save_path=None):

    # Reference trajectory
    gt = generate_trajectory(h0, t, amp, u_fun)

    # Integrate all three models
    models = [model_1, model_2, model_3]
    predictions = []

    for model in models:
        model.amp = amp
        model.u_fun = u_fun

        pred = odeint(model, h0, t, method="rk4", options={"step_size": dt})
        predictions.append(pred)

    pred1, pred2, pred3 = predictions

    # Extract first-dof displacement and velocity
    t_cpu = t.detach().cpu().numpy()

    x1_gt = gt[:, 0].detach().cpu().numpy()
    v1_gt = gt[:, 4].detach().cpu().numpy()

    x1_pred1 = pred1[:, 0].detach().cpu().numpy()
    v1_pred1 = pred1[:, 4].detach().cpu().numpy()

    x1_pred2 = pred2[:, 0].detach().cpu().numpy()
    v1_pred2 = pred2[:, 4].detach().cpu().numpy()

    x1_pred3 = pred3[:, 0].detach().cpu().numpy()
    v1_pred3 = pred3[:, 4].detach().cpu().numpy()

    # Compute acceleration directly from the governing right-hand sides
    rhs_gt = make_ground_truth_rhs(amp, u_fun)

    a1_gt, a1_pred1, a1_pred2, a1_pred3 = compute_a1_from_rhs(t, gt, pred1, pred2, pred3, rhs_gt, model_1, model_2, model_3)

    # Ensure acceleration arrays are NumPy arrays
    a1_gt = np.asarray(a1_gt)
    a1_pred1 = np.asarray(a1_pred1)
    a1_pred2 = np.asarray(a1_pred2)
    a1_pred3 = np.asarray(a1_pred3)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    prediction_styles = [
        {
            "values_x": x1_pred1,
            "values_v": v1_pred1,
            "values_a": a1_pred1,
            "label": label1,
            "color": "tab:red",
        },
        {
            "values_x": x1_pred2,
            "values_v": v1_pred2,
            "values_a": a1_pred2,
            "label": label2,
            "color": "tab:green",
        },
        {
            "values_x": x1_pred3,
            "values_v": v1_pred3,
            "values_a": a1_pred3,
            "label": label3,
            "color": "tab:blue",
        },
    ]

    # Displacement
    axes[0].plot(t_cpu, x1_gt, color="black", linewidth=1.8, label=r"Ground truth")
    for style in prediction_styles:
        axes[0].plot(t_cpu, style["values_x"], linestyle="--", linewidth=1.4, color=style["color"], label=style["label"])

    axes[0].set_ylabel(r"Displacement $x_1(t)$")
    axes[0].set_title(r"First-DOF displacement response", pad=8)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    # Velocity
    axes[1].plot(t_cpu, v1_gt, color="black", linewidth=1.8, label=r"Ground truth")
    for style in prediction_styles:
        axes[1].plot(t_cpu, style["values_v"], linestyle="--", linewidth=1.4, color=style["color"], label=style["label"])

    axes[1].set_ylabel(r"Velocity $\dot{x}_1(t)$")
    axes[1].set_title(r"First-DOF velocity response", pad=8)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    # Acceleration
    axes[2].plot(t_cpu, a1_gt, color="black", linewidth=1.8, label=r"Ground truth")
    for style in prediction_styles:
        axes[2].plot(t_cpu, style["values_a"], linestyle="--", linewidth=1.4, color=style["color"], label=style["label"])

    axes[2].set_xlabel(r"Time $t$ [s]")
    axes[2].set_ylabel(r"Acceleration $\ddot{x}_1(t)$")
    axes[2].set_title(r"First-DOF acceleration response", pad=8)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    # Shared legend below all subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return gt, pred1, pred2, pred3

def build_sindy_discrepancy_from_nn_exp2(h_true_list, model, device="cpu"):

    # Concatenate all trajectories
    # h_true = torch.cat(h_true_list, dim=0).to(device)   # (N,8)

    with torch.no_grad():
        a_disc_nn = model.mlp(h_true_list)                   # (N,4)

    X = h_true_list.detach().cpu().numpy()
    Y = a_disc_nn.detach().cpu().numpy()

    return X, Y


def run_sindy_discrepancy_from_nn_exp2(h_true_list,model,dt,threshold=0.01,device="cpu", t_train=1.0, t_test=1.0):


    # 1) Build X,Y from the trained neural net
    X, Y = build_sindy_discrepancy_from_nn_exp2(h_true_list=h_true_list,model=model,device=device)

    # 2) Same polynomial library as remainder of your code
    poly_lib = ps.PolynomialLibrary(degree=3,include_interaction=False,include_bias=False)

    # 3) Sparse optimizer
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=1000)

    print(f"\n===== SINDy discrepancy equations for model trained in {t_train} second in {t_test} seconds=====")

    sindy_models = []
    for i in range(4):
        y_i = Y[:, i:i+1]   # column i

        model_sindy = ps.SINDy(feature_library=poly_lib,optimizer=optimizer,)

        # Fit SINDy using NN-produced derivative y_i
        model_sindy.fit(X, t=dt, x_dot=y_i)
        sindy_models.append(model_sindy)

        coeffs = model_sindy.coefficients()[0]
        feats  = model_sindy.get_feature_names()

        print(f"\n--- dv{i+1}_disc (NN) ---")
        terms = []
        for c, term in zip(coeffs, feats):
            if abs(c) > 1e-9:
                terms.append(f"{c:+.5f} * {term}")
        if not terms:
            print("   0")
        else:
            print("   dv{} = ".format(i+1) + " ".join(terms))

    return sindy_models




if __name__ == "__main__":

    # Forcing distribution applied only at DOF 1
    B_force = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Amplitude levels
    amplitudes_train = [1.0, 2.0, 5.0, 10.0]


    # Parameters (the same as previous experiment)
    k1 = k2 = k3 = k4 = 10.0
    c1 = c2 = c3 = c4 = 0.5
    m1 = m2 = m3 = m4 = 1.0
    k_nl_true = 2.0  # cubic nonlinearity

    # Mass matrix (all ones -> identity)
    M = torch.eye(4, device=device)

    # Stiffness matrix K for a 4-DOF shear building (fixed base)
    K = torch.tensor([
        [k1 + k2,   -k2,       0.0,      0.0],
        [-k2,       k2 + k3,  -k3,       0.0],
        [0.0,      -k3,       k3 + k4,  -k4],
        [0.0,       0.0,      -k4,       k4]
    ], dtype=torch.float32, device=device)

    # Damping matrix (diagonal with c1..c4)
    C = torch.diag(torch.tensor([c1, c2, c3, c4], dtype=torch.float32, device=device))

    # Initial conditions for this experiment
    h0_1 = torch.tensor([ 2.0,  0.0, 0.0, 0.0,  0.0, -2.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_2 = torch.tensor([-2.0,  0.0, 0.0, 3.0, -2.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_3 = torch.tensor([ 0.0,  4.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_4 = torch.tensor([ 0.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Only one IC
    h0_train = [h0_2]

    # Time grids
    dt = 0.01  # 100 Hz
    t_train_end = 10.0
    t_full_end = 20.0
    t_test_end = 20.0

    t_train = torch.linspace(0.0, t_train_end, int(t_train_end / dt) + 1, device=device)
    t_test  = torch.linspace(0.0, t_test_end,  int(t_test_end  / dt) + 1, device=device)
    t_full  = torch.linspace(0.0, t_test_end,  int(t_full_end  / dt) + 1, device=device)

    print("Generating ground truth training data (white-noise excitation)...")
    traj_list = []  # shape [N_ic][n_amps]
    traj_list_full = []  # shape [N_ic][n_amps]

    traj_list, traj_list_full = generate_trajectory_lists(h0_train, t_train, t_full, amplitudes_train, u_base_fun)

    # save path
    save_path = "../../../models/"


    # Scheme 3
    save_model_path = save_path + "forced_vibration_model_h0_2_4sec_full_physics.pth"
    model = PINODEFuncForcedVibration(K,C,B_force)
    if os.path.exists(save_model_path):
        state = torch.load(save_model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    # model, loss_history, loss_history_20 = train_scheme_with_disc_multiIC_multiamp_sequential(
    #   scheme=3,
    #   model=model,
    #   h0_list=h0_train,
    #   amplitudes=amplitudes_train,
    #   traj_list=traj_list,
    #   traj_list_full=traj_list_full,
    #   t_train=t_train,
    #   num_epochs=2000,
    #   u_fun=u_base_fun,
    #   lr=1e-3,
    #   loss_log_path = loss_log_path,
    #   loss_history_init = [],
    #   loss_log_path_20 = loss_log_path_20,
    #   loss_history_20_init = [],
    #   save_model_path = save_model_path
    # )
    # save_model(model3, save_model_path)

    amp_test = 10.0
    traj_gt = generate_trajectory(h0_train[0], t_test, amp_test, u_base_fun)

    # Model with no known physics integrated trained on 0-10sec
    model_10_nophysics = NODEFuncNoPhysics()
    state = torch.load(save_path + "forced_vibration_model_h0_2_10sec_no_physics.pth", map_location=device, weights_only=True)
    model_10_nophysics.load_state_dict(state, strict=False)

    # Model with linear physics embedded and trained on 0-4sec
    model_4 = PINODEFuncForcedVibration(K, C, B_force)
    state = torch.load(save_path + "forced_vibration_model_h0_2_4sec_full_physics.pth", map_location=device, weights_only=True)
    model_4.load_state_dict(state, strict=False)

    # Model with linear physics embedded and trained on 0-6sec
    model_6 = PINODEFuncForcedVibration(K, C, B_force)
    state = torch.load(save_path + "forced_vibration_model_h0_2_6sec_full_physics.pth", map_location=device, weights_only=True)
    model_6.load_state_dict(state, strict=False)

    # Model with linear physics embedded and trained on 0-10sec
    model_10 = PINODEFuncForcedVibration(K, C, B_force)
    state = torch.load(save_path + "forced_vibration_model_h0_2_10sec_full_physics.pth", map_location=device, weights_only=True)
    model_10.load_state_dict(state, strict=False)

    evaluate_scheme_on_amp("No Physics 10sec Training", model_10_nophysics , amp_test, traj_gt)
    evaluate_scheme_on_amp("Full Physics 4sec Training", model_4 , amp_test, traj_gt)
    evaluate_scheme_on_amp("Full Physics 6sec Training", model_6 , amp_test, traj_gt)
    evaluate_scheme_on_amp("Full Physics 10sec Training", model_10 , amp_test, traj_gt)

    save_plot_path = "../../../logs/white_noise_forcing_models_evaluation.png"

    traj_gt, pred_test4, pred_test6, pred_test10 = plot_x1_v1_a1_three_preds(t_test, h0_train[0], amp_test, u_base_fun, model_4, model_6, model_10, dt=0.01, label1="4sec",label2="6sec",label3="10sec",save_path=save_plot_path)

    models_s3_nn = run_sindy_discrepancy_from_nn_exp2(h_true_list=pred_test10,model=model_10,dt=dt,threshold=0.5, device=device, t_train = 10.0, t_test = t_test_end)

    # Use IC 0 and amplitude 10.0 -> last amplitude of training
    h0 = h0_train[0]
    dt = 0.01
    t_test_end = 20
    t_test  = torch.linspace(0.0, t_test_end,  int(t_test_end  / dt) + 1, device=device)
    save_plot_path = "../../../logs/white_noise_exp2_models_evaluation_4_6_10_diff_fun.png"

    traj_gt_test, pred_test4, pred_test6, pred_test10 = plot_x1_v1_a1_three_preds(t_test, h0_train[0], amp_test, u_fun_cos, model_4, model_6, model_10, dt=0.01, label1="4sec",label2="6sec",label3="10sec",save_path=save_plot_path)

    models_s3_nn = run_sindy_discrepancy_from_nn_exp2(h_true_list=pred_test10,model=model_10,dt=dt,threshold=0.5, t_train = 10.0, t_test = t_test_end, device=device)

    # Another example is using the elcentro earthquake
    data = np.loadtxt("../../../data/elcentro.dat")

    dt_ec = 0.02  # time step 0.02s (same as the file)

    if data.ndim == 1:
        acc_ec = data # if it has one column, we assume that it represents the ground acceleration
        t_ec = np.arange(len(acc_ec)) * dt_ec # time is constructed
    else:
        t_ec = data[:, 0]
        acc_ec = data[:, 1]

    t_ec_torch   = torch.tensor(t_ec,    dtype=torch.float32, device=device)
    acc_ec_torch = torch.tensor(acc_ec,  dtype=torch.float32, device=device)

    u_fun_elcentro = make_timeseries_fun(t_ec_torch, acc_ec_torch)

    # Use the same dt and horizon as the record
    t_test_end = 40
    t_test  = torch.linspace(0.0, t_test_end,  int(t_test_end  / dt_ec) + 1, device=device)
    h0 = h0_train[0]     # since models were trained on this IC
    amp_test = 10
    save_plot_path = "../../../logs/white_noise_exp2_models_evaluation_4_6_10_elcentro.png"
    traj_gt_test, pred_test4, pred_test6, pred_test10 = plot_x1_v1_a1_three_preds(t_test, h0_train[0], amp_test, u_fun_cos, model_4, model_6, model_10, dt=0.02, label1="4sec",label2="6sec",label3="10sec",save_path=save_plot_path)

    models_s3_nn = run_sindy_discrepancy_from_nn_exp2(h_true_list=pred_test10,model=model_10,dt=dt,threshold=0.5, t_train = 10.0, t_test = t_test_end, device=device)

