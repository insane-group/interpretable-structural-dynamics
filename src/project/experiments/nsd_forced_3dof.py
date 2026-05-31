import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import time


from project.models.excitation import UFunFromSamples
from project.models.physics_only_3dof import PhysicsOnly3DOF
from project.models.pinode_linear_3dof import PINODEFuncLinear3DOF
from project.models.truth_linear_3dof import TruthLinear3DOF
from project.models.pinode_nsd_3dof import PINODEFuncNSD_3DOF
from project.models.truth_nsd_3dof import TruthPhaseNSD_3DOF
from project.models.nsd_net import NSD_Net

def model_accel(model, t, x, v):
    h = torch.cat([x, v], dim=-1)
    dh = model(t, h)
    ndof = x.shape[-1]
    return dh[..., ndof:]

def read_at2(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()

        parts = lines[3].split()
        npts = int(parts[0])
        dt_file = float(parts[1])

        data_str = " ".join(lines[4:])
        acc = np.fromstring(data_str, sep=" ")
        acc = acc[:npts]  # safety
        return acc.astype(np.float32), dt_file, npts


def rollout_central_difference(model, h0, t_grid):
    dt = (t_grid[1] - t_grid[0]).to(h0)
    T = t_grid.numel()

    # batchify
    if h0.ndim == 1:
        h0 = h0.unsqueeze(0)
        batch = False
    else:
        batch = True

    B, D = h0.shape
    ndof = D // 2

    x0 = h0[:, :ndof]
    v0 = h0[:, ndof:]

    # estimate x_{-1}
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


def rollout_newmark(model, h0, t_grid, beta=0.25, gamma=0.5, n_iter=3):

    dt = (t_grid[1] - t_grid[0]).to(h0)
    T = t_grid.numel()

    # batchify
    if h0.ndim == 1:
        h0 = h0.unsqueeze(0)
        batch = False
    else:
        batch = True

    B, D = h0.shape
    ndof = D // 2

    x = h0[:, :ndof]
    v = h0[:, ndof:]

    # initial acceleration from the model at t0
    t0 = t_grid[0]
    a = model_accel(model, t0, x, v)  # (B, ndof)

    traj = torch.zeros((T, B, D), device=h0.device, dtype=h0.dtype)
    traj[0, :, :ndof] = x
    traj[0, :, ndof:] = v

    dt2 = dt * dt
    c1 = (0.5 - beta) * dt2
    c2 = (1.0 - gamma) * dt

    for n in range(T - 1):
        tn1 = t_grid[n + 1]

        # Predictor
        x_pred = x + dt * v + c1 * a
        v_pred = v + c2 * a

        # Fixed-point iterations to resolve dependence on a_{n+1}(x_{n+1}, v_{n+1})
        x_next = x_pred
        v_next = v_pred
        a_next = a  # init

        for _ in range(n_iter):
            # acceleration at n+1 using current guesses
            a_next = model_accel(model, tn1, x_next, v_next)

            # Corrector
            x_next = x_pred + beta * dt2 * a_next
            v_next = v_pred + gamma * dt * a_next

        traj[n + 1, :, :ndof] = x_next
        traj[n + 1, :, ndof:] = v_next

        # advance
        x, v, a = x_next, v_next, a_next

    return traj if batch else traj[:, 0, :]

def train_scheme_with_disc_multiamp_exp3(
    scheme,
    model,
    h0,
    traj_true_list,   # list of tensors, one per amp, each (T,6) or (T,B,6)
    t_train,
    u_grid,           # forcing samples on t_train
    amp_list,         # list of amplitudes, same length/order as traj_true_list
    num_epochs=4000,
    lr=1e-3,
    log_interval=20,
    loss_log_path=None,
    loss_history_init=None,
    save_model_path=None,
    grad_clip_norm=1.0,
    power_penalty_lambda=0.0,
    elapsed_time_prev=0.0,
    device="cpu",
):

    assert len(amp_list) == len(traj_true_list), "amp_list and traj_true_list must have same length"

    model = model.to(device)
    model.train()

    # forcing is the same time series; only amp changes
    model.u_fun = UFunFromSamples(t_train, u_grid.to(device))

    opt = optim.Adam(model.parameters(), lr=lr)

    # --------- resume logic ----------
    loss_history_list = []
    global_epoch = 0
    if loss_history_init is not None:
        loss_history_list = loss_history_init.tolist() if isinstance(loss_history_init, np.ndarray) else list(loss_history_init)
        if len(loss_history_list) > 0:
            try:
                global_epoch = int(loss_history_list[-1][0])
            except Exception:
                global_epoch = 0
    # -------------------------------

    start_time = time.time()
    first_print = False

    T = len(t_train)

    print(f"\n=== Training MULTI-AMP (ALL amps per epoch), scheme {scheme}, for {num_epochs} epochs ===")
    print(f"amps={amp_list} | power_penalty_lambda={power_penalty_lambda}")

    for local_epoch in range(1, num_epochs + 1):
        global_epoch += 1
        opt.zero_grad(set_to_none=True)

        total_loss = 0.0

        # diagnostics containers
        per_amp_stats = []          # (amp, Lx, Lv, Ltot)

        # for hinge evaluation we'll reuse the last pred per amp (no grad)
        preds_cache = {}

        for amp, traj_true in zip(amp_list, traj_true_list):
            amp = float(amp)
            traj_true = traj_true.to(device)

            # set current amplitude
            model.amp = amp

            # shape checks
            assert traj_true.shape[0] == T, f"traj_true first dim must be T={T}, got {traj_true.shape[0]}"
            state_dim = traj_true.shape[-1]
            ndof = state_dim // 2

            x_true = traj_true[..., :ndof]
            v_true = traj_true[..., ndof:]

            # ---- per-amp normalization ----
            eps = 1e-8
            reduce_dims = tuple(range(traj_true.ndim - 1))
            sx = x_true.std(dim=reduce_dims, unbiased=False) + eps
            sv = v_true.std(dim=reduce_dims, unbiased=False) + eps
            while sx.ndim < x_true.ndim:
                sx = sx.unsqueeze(0)
                sv = sv.unsqueeze(0)

            # rollout for this amp
            pred = rollout_central_difference(model, h0, t_train)
            assert pred.shape == traj_true.shape, f"pred.shape={pred.shape}, traj_true.shape={traj_true.shape}"

            x_pred = pred[..., :ndof]
            v_pred = pred[..., ndof:]

            loss_x = torch.mean(((x_pred - x_true) / sx) ** 2)
            loss_v = torch.mean(((v_pred - v_true) / sv) ** 2)
            loss_amp = loss_x + loss_v

            total_loss = total_loss + loss_amp
            per_amp_stats.append(
                (amp,
                 float(loss_x.detach().cpu()),
                 float(loss_v.detach().cpu()),
                 float(loss_amp.detach().cpu()))
            )

            preds_cache[amp] = (pred.detach(), traj_true.detach())

        # average across amps
        total_loss = total_loss / len(amp_list)

        total_loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        opt.step()

        if global_epoch % log_interval == 0:
            loss_history_list.append([global_epoch, float(total_loss.item())])

        if (not first_print) or (global_epoch % 10 == 0):
            first_print = True
            end_time = time.time() - start_time + elapsed_time_prev

            print(
                f"[S{scheme}] Global {global_epoch:5d} | "
                f"Total(avg over amps)={total_loss.item():.3e} | Time={end_time/60:.3f} min"
            )
            for amp, lx, lv, ltot in per_amp_stats:
                print(f"   amp={amp:6.1f} | Lx={lx:.3e}, Lv={lv:.3e}, Total={ltot:.3e}")

            if save_model_path is not None:
                torch.save(model.state_dict(), save_model_path)
                print(f"Saved model to {save_model_path}")

        if loss_log_path is not None and (local_epoch % 40 == 0):
            np.save(loss_log_path, np.array(loss_history_list, dtype=np.float32))
            print(f"Saved partial loss history to {loss_log_path}")

    loss_history_arr = np.array(loss_history_list, dtype=np.float32)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
        print(f"Saved model to {save_model_path}")

    if loss_log_path is not None:
        np.save(loss_log_path, loss_history_arr)
        print(f"\nSaved loss history to: {loss_log_path}")

    return model, loss_history_arr

def plot_trajectories(tt,T_train_sec,trajectories,labels,idx,ylabel,save_path=None):

    fig = plt.figure(figsize=(10, 8))

    for traj, label in zip(trajectories, labels):
        if torch.is_tensor(traj):
            y = traj[:, idx].detach().cpu().numpy()
        else:
            y = np.asarray(traj)[:, idx]

        plt.plot(tt, y, label=label)

    plt.axvline(T_train_sec, linestyle="--", label="train end")

    plt.xlabel("t [s]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.show()

def compute_matrix_mismatch(model):
    # layer 1: Linear(6->10), layer 2: Linear(10->3)
    L1 = model.mlp[0]
    L2 = model.mlp[1]

    with torch.no_grad():
        W1 = L1.weight.detach()   # (10, 6)
        b1 = L1.bias.detach()     # (10,)
        W2 = L2.weight.detach()   # (3, 10)
        b2 = L2.bias.detach()     # (3,)

        # Effective map: a_disc = W_eff h + b_eff
        W_eff = W2 @ W1                  # (3, 6)
        b_eff = b2 + W2 @ b1             # (3,)


    Wx = W_eff[:, :3]   # (3,3) multiplies x
    Wv = W_eff[:, 3:]   # (3,3) multiplies v


    with torch.no_grad():
        K_upd = K - (M @ Wx)   # (3,3)
        C_upd = C - (M @ Wv)   # (3,3)

    return K_upd, C_upd

def nsd_force_bilinear(x1, d0=0.15, k_post=-10.0):

    # amount beyond threshold
    excess = torch.relu(torch.abs(x1) - d0)
    return k_post * excess * torch.sign(x1)



if __name__ == "__main__":


    device = "cpu"
    print("Using device:", device)

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

    r = torch.ones(3, device=device)          # participation vector
    B = -(M @ r)      # (3,)

    # Initial conditions for this experiment)
    h0_1 = torch.tensor([ 2.0,  0.0, 0.0, 0.0,  0.0, -2.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_2 = torch.tensor([-2.0,  0.0, 0.0, -2.0,  0.0, 0.0], dtype=torch.float32, device=device)
    h0_3 = torch.tensor([ 0.0,  4.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_4 = torch.tensor([ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    h0_train = [h0_2]

    # Read AT2 and resample to dt_model = 1/256 over the full physical duration

    at2_path = "../../../data/kobe.at2"

    u_file_np, dt_file, npts = read_at2(at2_path)

    # File's physical timeline (dt_file = 0.01, NPTS=3200 -> ~31.99s)
    T_end = (npts - 1) * dt_file
    t_file = torch.arange(0.0, T_end + dt_file, dt_file)   # length npts

    # Integration grid
    dt = 1.0 / 256.0
    t_full = torch.arange(0.0, T_end + dt, dt, device=device)  # ~8193 points

    # Resample acceleration onto t_full (linear interpolation)
    u_full_np = np.interp(t_full.detach().cpu().numpy(),
                        t_file.numpy(),
                        u_file_np).astype(np.float32)

    u_full = torch.tensor(u_full_np, device=device)


    T_train_sec = 15.0
    N_train = int(T_train_sec / dt) + 1
    t_train = t_full[:N_train]
    u_train = u_full[:N_train]

    # Start from nominal
    M_true = M.clone()
    K_true = K.clone()
    C_true = C.clone()

    # Add controlled mismatch
    K_true = 0.97 * K_true  # stiffness is 3% lower than nominal
    C_true = 1.25 * C_true  # damping is 25% higher than nominal

    truth_amp_full = []
    truth_amp_train = []
    amp_list = [50,100,150]
    for amp in amp_list:

        truth = TruthLinear3DOF(M_true, K_true, C_true, amp).to(device)
        truth.u_fun = UFunFromSamples(t_full, u_full)

        # define forcing function on the full horizon
        u_fun_full = UFunFromSamples(t_full, u_full)

        with torch.no_grad():
            # traj_meas_full = odeint(truth, h0_4, t_full, method="rk4")  # (T_full, 6)
            traj_meas_full = rollout_central_difference(truth, h0_4, t_full)

        traj_meas_train = traj_meas_full[:N_train]  # (T_train, 6)

        truth_amp_full.append(traj_meas_full)
        truth_amp_train.append(traj_meas_train)


    model = PINODEFuncLinear3DOF(M, K, C,B)

    save_path = "../../../models/"

    save_model_path = save_path + "linear_with_forced_vibration_base_model_15sec.pth"
    state = torch.load(save_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)


    # model, loss_hist = train_scheme_with_disc_multiamp_exp3(
    #     scheme=3,
    #     model=model,
    #     h0=h0_4,
    #     traj_true_list=truth_amp_train,
    #     t_train=t_train,
    #     u_grid=u_train,
    #     amp_list=amp_list,
    #     num_epochs=300,
    #     lr=5e-3,
    #     log_interval=20,
    #     loss_log_path=[],
    #     save_model_path=save_model_path,
    #     loss_history_init=None,
    # )

    # --- set the forcing on the full horizon ---
    model.eval()
    model.u_fun = UFunFromSamples(t_full, u_full)
    model.amp = 100.0

    # --- extrapolate ---
    with torch.no_grad():
        pred_full = rollout_central_difference(model, h0_4, t_full)  # (N_full, 6)

    # pred_full contains [x1 x2 x3 v1 v2 v3]
    x_pred_full = pred_full[:, :3]
    v_pred_full = pred_full[:, 3:]

    phys_only = PhysicsOnly3DOF(M, K, C, B, t_full, u_full, amp=100.0).to(device)

    with torch.no_grad():
        traj_phys_full = rollout_central_difference(phys_only, h0_4, t_full)  # (T_full,6)

    save_plot_path = "../../../logs/"
    save_path_x = save_plot_path + "linear_3dof_100_full_traj_x1.png"
    save_path_v = save_plot_path + "linear_3dof_100_full_traj_v1.png"

    plot_trajectories(t_full, T_train_sec, [truth_amp_full[1], traj_phys_full, pred_full], labels=["measured (synthetic truth)", "physics only", "NN + physics"], idx=0, ylabel="x1", save_path = save_path_x)
    plot_trajectories(t_full, T_train_sec, [truth_amp_full[1], traj_phys_full, pred_full], labels=["measured (synthetic truth)", "physics only", "NN + physics"], idx=3, ylabel="v1", save_path = save_path_v)


    K_upd, C_upd = compute_matrix_mismatch(model)

    print("K_true:\n", K_true)
    print("K_upd:\n", K_upd)
    print("K:\n", K)

    print("C_true:\n", C_true)
    print("C_upd:\n", C_upd)
    print("C:\n", C)

    truth2_amp_full = []
    truth2_amp_train = []
    amp_list = [50,100,150]
    for amp in amp_list:

        truth2 = TruthPhaseNSD_3DOF(M_true, K_true, C_true, nsd_force_bilinear).to(device)
        truth2.u_fun = UFunFromSamples(t_full, u_full)
        truth2.amp=amp

        # define forcing function on the full horizon
        u_fun_full = UFunFromSamples(t_full, u_full)

        with torch.no_grad():
            traj2_meas_full = rollout_central_difference(truth2, h0_4, t_full)

        traj2_meas_train = traj2_meas_full[:N_train]  # (T_train, 6)

        truth2_amp_full.append(traj2_meas_full)
        truth2_amp_train.append(traj2_meas_train)

    model.u_fun = UFunFromSamples(u_train, t_train)
    nn2 = NSD_Net().to(device)
    model_nsd = PINODEFuncNSD_3DOF(M, K, C, model.mlp, nn2).to(device)

    # freeze NN1
    for p in model_nsd.mlp.parameters():
        p.requires_grad = False
    model_nsd.mlp.eval()

    save_model_path = save_path + "nsd_forced_vibration_base_15sec_model.pth"

    state = torch.load(save_model_path, map_location=device, weights_only=True)
    model_nsd.load_state_dict(state, strict=False)

    # elapsed_time_prev = 0

    # model_nsd, loss_hist = train_scheme_with_disc_multiamp_exp3(
    #     scheme=3,
    #     model=model_nsd2,
    #     h0=h0_4,
    #     traj_true_list=truth2_amp_train,
    #     t_train=t_train,
    #     u_grid=u_train,
    #     amp_list=amp_list,
    #     num_epochs=1000,
    #     lr=5e-3,
    #     log_interval=20,
    #     loss_log_path=loss_log_path,
    #     save_model_path=save_model_path,
    #     loss_history_init=[],
    #     grad_clip_norm=1.0,
    #     elapsed_time_prev = elapsed_time_prev
    # )

    model.eval()
    model.u_fun = UFunFromSamples(t_full, u_full)
    model.amp = 100.0

    # --- extrapolate ---
    with torch.no_grad():
        pred_train = odeint(model, h0_4, t_full, method="rk4")  # (N_full, 6)

    # --- set the forcing on the full horizon ---
    model_nsd.eval()
    model_nsd.u_fun = UFunFromSamples(t_full, u_full)
    model_nsd.amp = 100.0

    # --- extrapolate ---
    with torch.no_grad():
        pred_train_nsd = odeint(model_nsd, h0_4, t_full, method="rk4")  # (N_full, 6)


    save_path_x = save_plot_path + "nsd_3dof_100_full_traj_x1.png"
    save_path_v = save_plot_path + "nsd_3dof_100_full_traj_v1.png"
    plot_trajectories(t_full, T_train_sec, [truth_amp_full[1], traj_phys_full, pred_full], labels=["measured (synthetic truth)", "NN1 + physics", "NN1 + NN2 + physics"], idx=0, ylabel="χ1", save_path = save_path_x)
    plot_trajectories(t_full, T_train_sec, [truth_amp_full[1], traj_phys_full, pred_full], labels=["measured (synthetic truth)", "NN1 + physics", "NN1 + NN2 + physics"], idx=3, ylabel="v1", save_path = save_path_v)


    data = np.loadtxt("../../../data/elcentro.dat")
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

    u_full_np_2 = np.interp(t_full.detach().cpu().numpy(),t_ec,acc_ec).astype(np.float32)
    u_full_2 = torch.tensor(u_full_np_2, device=device)

    T_train_sec = 15.0
    N_train = int(T_train_sec / dt) + 1
    t_train_2 = t_full[:N_train]
    u_train_2 = u_full[:N_train]

    # Run truth trajectory under El Centro
    true_trajectory = TruthPhaseNSD_3DOF(M_true, K_true, C_true, nsd_force_bilinear)

    true_trajectory.u_fun = UFunFromSamples(t_full, u_full_2)
    true_trajectory.amp = 120.0

    model.amp = 120.0
    model.u_fun = UFunFromSamples(t_full, u_full_2)

    model_nsd.amp = 120.0
    model_nsd.u_fun = UFunFromSamples(t_full, u_full_2)


    with torch.no_grad():
        traj_nn = odeint(model_nsd, h0_4, t_full, method="rk4", options={"step_size" : dt_ec})
        traj = odeint(true_trajectory, h0_4, t_full, method="rk4", options={"step_size" : dt_ec})  # (T,6)


    save_path_x = save_plot_path + "nsd_3dof_120_full_traj_x1_elcentro_eval.png"
    save_path_v = save_plot_path + "nsd_3dof_120_full_traj_v1_elcentro_eval.png"
    plot_trajectories(t_full, T_train_sec, [traj, traj_nn], labels=["measured (synthetic truth)", "pi-node"], idx=0, ylabel="x1", save_path = save_path_x)
    plot_trajectories(t_full, T_train_sec, [traj, traj_nn], labels=["measured (synthetic truth)", "pi-node"], idx=3, ylabel="v1", save_path = save_path_v)

