# Import regular libs
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
import pysindy as ps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import models
from project.models.pinode_free_4dof import PINODEFunc4DOF


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
    "ps.fonttype": 42,
})

def ground_truth_rhs(t, h):

    x = h[:4]
    v = h[4:]

    # Linear restoring + damping: -Kx - Cv
    lin_force = -K @ x - C @ v

    # Cubic nonlinearity on DOF 1 only
    f_nl = torch.zeros_like(x)
    f_nl[0] = -k_nl_true * x[0]**3

    # Total acceleration (M = I)
    a = lin_force + f_nl

    dh = torch.zeros_like(h)
    dh[:4] = v      # x' = v
    dh[4:] = a      # v' = a
    return dh

@torch.no_grad()
def generate_trajectory(h0, t_grid):

    # This is the reference trajectory
    sol = odeint(ground_truth_rhs, h0, t_grid, method='rk4')  # (T, 8)
    return sol

def discrepancy_reference(x, v, scheme, K, C, k_nl):
    # x, v: (T,4)

    # Linear part in standard form: Kx + Cv
    lin = torch.matmul(x, K.T) + torch.matmul(v, C.T)

    # Nonlinear force only on DOF1: k_nl * x1^3
    f_nl = torch.zeros_like(x)
    f_nl[:, 0] = k_nl * x[:, 0]**3

    # True acceleration according to full physics: a_true = -(Kx + Cv + f_nl)
    a_true = -(lin + f_nl)

    # Depending on the scheme, discrepancy is defined differently
    if scheme == 1:
        # Scheme 1 has no physics in v', so discrepancy = full true acceleration
        a_disc = a_true
    elif scheme == 2:
        # Scheme 2 uses a_phy = -0.7(Kx + Cv)
        a_phy = -0.7 * lin
        # discrepancy = a_true - a_phy = missing 0.3*lin - f_nl
        a_disc = a_true - a_phy
    elif scheme == 3:
        # Scheme 3 uses full linear physics a_phy = -lin
        a_phy = -lin
        # discrepancy = a_true - a_phy = -f_nl (pure nonlinear)
        a_disc = a_true - a_phy
    else:
        raise ValueError("scheme must be 1, 2, or 3")

    return a_disc

def true_discrepancy_on_traj(scheme, traj):
    # traj: (T,8) [x(4), v(4)] along a TRUE trajectory
    x = traj[..., :4]
    v = traj[..., 4:]
    # This returns the analytic discrepancy a_disc(T,4) for that scheme
    return discrepancy_reference(x, v, scheme=scheme, K=K, C=C, k_nl=k_nl_true)

def train_scheme_with_disc(scheme, model, num_epochs=200, lr=1e-3, lambda_disc=1.0):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    epochs_ic1 = num_epochs // 4
    epochs_ic2 = num_epochs - epochs_ic1

    # Phase 1: train only on IC1 for 1250 epochs
    for epoch in range(1, epochs_ic1):
        opt.zero_grad()

        # Model trajectory for IC1
        pred1 = odeint(model, h0_1, t_train, method='rk4')

        # Data loss for IC1 only
        loss_state = mse(pred1, traj1_train)

        # Discrepancy supervision on TRUE traj1
        with torch.no_grad():
            disc1_true = true_discrepancy_on_traj(scheme, traj1_train)

        disc1_nn = model.mlp(traj1_train)
        loss_disc = mse(disc1_nn, disc1_true)

        loss = loss_state + lambda_disc * loss_disc
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{scheme}] Epoch {epoch:4d}, loss = {loss.item():.4e}")

    # Phase 2: train only on IC2 for 3750 epochs
    for epoch in range(1, epochs_ic2):
        opt.zero_grad()

        # Model trajectory for IC2
        pred2 = odeint(model, h0_2, t_train, method='rk4')

        # Data loss for IC2 only
        loss_state = mse(pred2, traj2_train)

        # Discrepancy supervision on TRUE traj2
        with torch.no_grad():
            disc2_true = true_discrepancy_on_traj(scheme, traj2_train)

        disc2_nn = model.mlp(traj2_train)
        loss_disc = mse(disc2_nn, disc2_true)

        loss = loss_state + lambda_disc * loss_disc
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{scheme}] Epoch {epoch:4d}, loss = {loss.item():.4e}")

    return model


@torch.no_grad()
def evaluate_scheme(scheme_name, model):

    loss_fn = nn.MSELoss()

    # IC2, 0–12s
    pred2 = odeint(model, h0_2, t_test, method='rk4')
    mse2_full = loss_fn(pred2, traj2_test).item()

    # split at 6s (same dt as train)
    mid_idx = len(t_train)
    mse2_0_6  = loss_fn(pred2[:mid_idx], traj2_test[:mid_idx]).item()
    mse2_6_12 = loss_fn(pred2[mid_idx:], traj2_test[mid_idx:]).item()

    # IC3, 0–12s
    pred3 = odeint(model, h0_3, t_test, method='rk4')
    mse3_full = loss_fn(pred3, traj3_test).item()
    mse3_0_6  = loss_fn(pred3[:mid_idx], traj3_test[:mid_idx]).item()
    mse3_6_12 = loss_fn(pred3[mid_idx:], traj3_test[mid_idx:]).item()

    print(f"\n=== Evaluation for {scheme_name} ===")
    print(f"IC2 (0–12s) MSE total: {mse2_full:.4e}")
    print(f"IC2 (0–6s)  MSE:       {mse2_0_6:.4e}")
    print(f"IC2 (6–12s) MSE:       {mse2_6_12:.4e}")
    print(f"IC3 (0–12s) MSE total: {mse3_full:.4e}")
    print(f"IC3 (0–6s)  MSE:       {mse3_0_6:.4e}")
    print(f"IC3 (6–12s) MSE:       {mse3_6_12:.4e}")


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def plot_nn_vs_interdofs(models, scheme, h_true_list, K_true, C_true, k_nl_true, device="cpu", save_path=None):

    if scheme not in (1, 2, 3):
        raise ValueError(f"scheme must be 1, 2, or 3, got {scheme}")

    # Concatenate reference states
    h_true = torch.cat(h_true_list, dim=0).to(device)
    x = h_true[:, :4]
    v = h_true[:, 4:]

    # Relative displacements
    drift_1 = x[:, 0]
    drift_2 = x[:, 1] - x[:, 0]
    drift_3 = x[:, 2] - x[:, 1]
    drift_4 = x[:, 3] - x[:, 2]

    drift_list = [drift_4, drift_3, drift_2, drift_1]

    drift_labels = [
        r"Relative displacement $x_4-x_3$",
        r"Relative displacement $x_3-x_2$",
        r"Relative displacement $x_2-x_1$",
        r"Displacement $x_1$",
    ]

    row_dof_idx = [3, 2, 1, 0]

    scheme_titles = {
        1: r"Scheme 1: No physics",
        2: r"Scheme 2: Weak physics",
        3: r"Scheme 3: Full linear physics",
    }

    scheme_colors = {
        1: "tab:red",
        2: "tab:blue",
        3: "tab:green",
    }

    model = models[scheme].to(device)
    model.eval()

    with torch.no_grad():
        nn_output = model.mlp(h_true).detach().cpu().numpy()

    reference_discrepancy = discrepancy_reference(x, v, scheme=scheme, K=K_true.to(device), C=C_true.to(device), k_nl=k_nl_true).detach().cpu().numpy()

    drift_np = [drift.detach().cpu().numpy() for drift in drift_list]

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), squeeze=False)
    axes = axes.flatten()

    for row, ax in enumerate(axes):
        dof_idx = row_dof_idx[row]

        x_axis = drift_np[row]
        neural_values = nn_output[:, dof_idx]
        reference_values = reference_discrepancy[:, dof_idx]

        sort_idx = np.argsort(x_axis)

        ax.plot(x_axis[sort_idx], reference_values[sort_idx], color="black", linewidth=1.8, label=r"Analytical discrepancy")

        ax.plot(x_axis[sort_idx], neural_values[sort_idx], linestyle="--", linewidth=1.6, color=scheme_colors[scheme], label=r"Neural discrepancy")

        ax.set_ylim(-40, 60)
        ax.set_xlabel(drift_labels[row])
        ax.set_ylabel(rf"$\Delta \ddot{{x}}_{dof_idx + 1}$")

        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(scheme_titles[scheme], fontsize=13, y=0.995)

    # Shared legend below the full figure
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_ic_results(t, gt, pred1, pred2, pred3, quantity="displacement", save_path=None):

    if quantity not in ("displacement", "velocity"):
        raise ValueError(
            f"quantity must be 'displacement' or 'velocity', got '{quantity}'"
        )

    time_np = t.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    pred1_np = pred1.detach().cpu().numpy()
    pred2_np = pred2.detach().cpu().numpy()
    pred3_np = pred3.detach().cpu().numpy()

    if quantity == "displacement":
        state_indices = [0, 1, 2, 3]
        state_labels = [
            r"Displacement $x_1(t)$",
            r"Displacement $x_2(t)$",
            r"Displacement $x_3(t)$",
            r"Displacement $x_4(t)$"
        ]
        ylabel = r"Displacement"
    else:
        state_indices = [4, 5, 6, 7]
        state_labels = [
            r"Velocity $\dot{x}_1(t)$",
            r"Velocity $\dot{x}_2(t)$",
            r"Velocity $\dot{x}_3(t)$",
            r"Velocity $\dot{x}_4(t)$"
        ]
        ylabel = r"Velocity"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, state_idx, state_label in zip(axes, state_indices, state_labels):
        ax.plot(time_np, gt_np[:, state_idx], color="black", linewidth=1.8, label=r"Ground truth")
        ax.plot(time_np, pred1_np[:, state_idx], linestyle="--", linewidth=1.4, color="tab:red", label=r"Scheme 1: No physics")
        ax.plot(time_np, pred2_np[:, state_idx], linestyle="--", linewidth=1.4, color="tab:blue", label=r"Scheme 2: Weak physics")
        ax.plot(time_np, pred3_np[:, state_idx], linestyle="--", linewidth=1.4, color="tab:green", label=r"Scheme 3: Full linear physics")

        ax.set_title(state_label, pad=8)
        ax.set_xlabel(r"Time $t$ [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    # One shared legend below all subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def build_sindy_discrepancy_from_nn(h_true_list, model, device="cpu"):

    # Combine all TRUE trajectories (from training or test)
    # into a single dataset that SINDy will analyze
    h_true = torch.cat(h_true_list, dim=0).to(device)  # shape (N,8)

    # Evaluate the neural network on the TRUE states
    with torch.no_grad():
        a_disc_nn = model.mlp(h_true)  # shape (N,4)

    # Convert to numpy for PySINDy
    X = h_true.cpu().numpy()     # full state
    Y = a_disc_nn.cpu().numpy()  # NN discrepancy

    return X, Y

def run_sindy_discrepancy_from_nn(h_true_list, model, dt, threshold=0.05, device="cpu"):


    # Build (X, Y) dataset from NN outputs
    X, Y = build_sindy_discrepancy_from_nn(h_true_list=h_true_list,model=model,device=device)
    # X: (N,8) state vectors
    # Y: (N,4) NN-generated discrepancy accelerations

    # Polynomial basis for SINDy (up to x^3, no cross terms)
    poly_lib = ps.PolynomialLibrary(degree=3,include_interaction=False,include_bias=False)

    # Sparse optimizer for selecting few active terms
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=100)

    print("\n===== SINDy discrepancy equations from NN (learned) =====")

    models = []  # to store one SINDy model per dv component

    # Fit four separate equations: dv1, dv2, dv3, dv4
    for i in range(4):
        y_i = Y[:, i:i+1]  # extract column i

        model_sindy = ps.SINDy(feature_library=poly_lib,optimizer=optimizer)

        # Fit SINDy with known dt and known derivative y_i
        model_sindy.fit(X, t=dt, x_dot=y_i)

        models.append(model_sindy)

        # Retrieve coefficients and feature names
        coeffs = model_sindy.coefficients()[0]
        feats  = model_sindy.get_feature_names()

        print(f"\n--- dv{i+1}_disc (NN) ---")
        terms = []
        for c, term in zip(coeffs, feats):
            # Print only nonzero terms -> sparse result
            if abs(c) > 1e-9:
                terms.append(f"{c:+.4f} * {term}")

        # If nothing survived thresholding -> output zero
        if len(terms) == 0:
            print("0")
        else:
            print("dv{} = ".format(i+1) + " ".join(terms))

    # Return all four DOF models for downstream coefficient comparison
    return models

if __name__ == "__main__":

    # Parameters
    k1 = k2 = k3 = k4 = 10.0
    c1 = c2 = c3 = c4 = 0.5
    m1 = m2 = m3 = m4 = 1.0
    k_nl_true = 2.0

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

    # Initial conditions examples (8D state: [x1,x2,x3,x4,v1,v2,v3,v4])
    h0_1 = torch.tensor([ 2.0,  0.0, 0.0, 0.0,  0.0, -2.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_2 = torch.tensor([-2.0,  0.0, 0.0, 3.0, -2.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    h0_3 = torch.tensor([ 0.0,  4.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Time grids
    dt = 0.01
    t_train_end = 6.0
    t_test_end = 12.0

    # Same dt for train/test
    t_train = torch.linspace(0.0, t_train_end, int(t_train_end / dt) + 1, device=device)
    t_test  = torch.linspace(0.0, t_test_end,  int(t_test_end  / dt) + 1, device=device)

    print("Generating ground truth data...")
    # Training trajectories (0–6s)
    traj1_train = generate_trajectory(h0_1, t_train)  # IC1
    traj2_train = generate_trajectory(h0_2, t_train)  # IC2

    # Test/generalization trajectories (0–12s)
    traj2_test = generate_trajectory(h0_2, t_test)    # IC2
    traj3_test = generate_trajectory(h0_3, t_test)    # IC3

    save_path = "../../../models/"

    # Load already-trained models
    model1 = PINODEFunc4DOF(K, C, scheme=1)
    state1 = torch.load(save_path + "free_vibration_model_no_physics.pth", map_location=device, weights_only=True)
    model1.load_state_dict(state1, strict=False)

    model2 = PINODEFunc4DOF(K, C, scheme=2)
    state2 = torch.load(save_path + "free_vibration_model_partial_physics.pth", map_location=device, weights_only=True)
    model2.load_state_dict(state2, strict=False)

    model3 = PINODEFunc4DOF(K, C, scheme=3)
    state3 = torch.load(save_path + "free_vibration_model_full_linear_physics.pth", map_location=device, weights_only=True)
    model3.load_state_dict(state3, strict=False)

    # model3 = train_scheme_with_disc(3, model3, num_epochs=5000, lr=1e-3, lambda_disc=1.0)

    models = {
        1: model1,  # trained PINODEFuncScheme1
        2: model2,  # trained PINODEFuncScheme2
        3: model3,  # trained PINODEFuncScheme3
    }

    # For evaluation
    h_true_list = [traj2_train]
    save_path1 = "../../../logs/free_vibration_inter_dof_4dof_scheme1.png"
    save_path2 = "../../../logs/free_vibration_inter_dof_4dof_scheme2.png"
    save_path3 = "../../../logs/free_vibration_inter_dof_4dof_scheme3.png"

    plot_nn_vs_interdofs(models=models, scheme=1, h_true_list=h_true_list, K_true=K, C_true=C, k_nl_true=k_nl_true, device=device, save_path=save_path1)
    plot_nn_vs_interdofs(models=models, scheme=2, h_true_list=h_true_list, K_true=K, C_true=C, k_nl_true=k_nl_true, device=device, save_path=save_path2)
    plot_nn_vs_interdofs(models=models, scheme=3, h_true_list=h_true_list, K_true=K, C_true=C, k_nl_true=k_nl_true, device=device, save_path=save_path3)

    # --- Evaluate IC2 plot (0–12s) ---

    with torch.no_grad():
        pred2_s1 = odeint(model1, h0_2, t_test, method='rk4')  # scheme 1
        pred2_s2 = odeint(model2, h0_2, t_test, method='rk4')  # scheme 2
        pred2_s3 = odeint(model3, h0_2, t_test, method='rk4')  # scheme 3

    save_path_x = "../../../logs/free_vibration_4dof_extrapolation_ic2_displacements.png"
    save_path_v = "../../../logs/free_vibration_4dof_extrapolation_ic2_velocities.png"

    # Plot full 12 seconds
    plot_ic_results(t_test, traj2_test, pred2_s1, pred2_s2, pred2_s3, quantity="displacement", save_path=save_path_x)
    plot_ic_results(t_test, traj2_test, pred2_s1, pred2_s2, pred2_s3, quantity="velocity", save_path=save_path_v)

    evaluate_scheme(1, model1)
    evaluate_scheme(2, model2)
    evaluate_scheme(3, model3)

    with torch.no_grad():
        pred3_s1 = odeint(model1, h0_3, t_test, method='rk4')  # scheme 1
        pred3_s2 = odeint(model2, h0_3, t_test, method='rk4')  # scheme 2
        pred3_s3 = odeint(model3, h0_3, t_test, method='rk4')  # scheme 3

    save_path_x = "../../../logs/free_vibration_4dof_evaluation_ic3_displacements.png"
    save_path_v = "../../../logs/free_vibration_4dof_evaluation_ic3_velocities.png"

    # Plot full 12 seconds
    plot_ic_results(t_test, traj3_test, pred3_s1, pred3_s2, pred3_s3, quantity="displacement", save_path=save_path_x)
    plot_ic_results(t_test, traj3_test, pred3_s1, pred3_s2, pred3_s3, quantity="velocity", save_path=save_path_v)

    models_s2_nn = run_sindy_discrepancy_from_nn(h_true_list=[traj2_train],model=model2,dt=dt,threshold=0.05,device=device)
    models_s3_nn = run_sindy_discrepancy_from_nn(h_true_list=[traj2_train],model=model3,dt=dt,threshold=0.05,device=device)




