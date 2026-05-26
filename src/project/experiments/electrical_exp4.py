import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import copy

from project.models.physics_informed_satmd import SATMDFrequencyRelativeParameterModel
from project.models.satmd_groundtruth import SATMDFrequencyGroundTruth
from project.models.rl_inverse_net import RLInverseNet

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


def plot_frf(groundtruth_model, relative_model, omega_t, freq_hz, save_path=None, dpi=600, figsize=(5.8, 3.4)):

    groundtruth_model.eval()
    relative_model.eval()

    with torch.no_grad():
        Y_true = groundtruth_model.solve_frf(omega_t).detach().cpu()
        Y_pred = relative_model.solve_frf(omega_t).detach().cpu()

    freq_hz = np.asarray(freq_hz)

    Y_true_mag = torch.abs(Y_true).numpy()
    Y_pred_mag = torch.abs(Y_pred).numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(freq_hz, Y_true_mag, label="Target", linewidth=1.5)

    ax.plot(freq_hz, Y_pred_mag, linestyle="--", label="Identified", linewidth=1.5)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$|Y(\omega)|$")

    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()

    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        directory = os.path.dirname(save_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if ext:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(root + ".pdf", bbox_inches="tight")
            fig.savefig(root + ".png", dpi=dpi, bbox_inches="tight")
            fig.savefig(root + ".svg", dpi=dpi, bbox_inches="tight")

    return fig, ax


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

def compute_frequency_sensitivity(model, acc_curve, norm_info, freq_hz):

    model.eval()

    # Convert the complex acceleration FRF into a real-valued feature vector
    # Then add a batch dimension
    feat = acceleration_to_features(acc_curve).unsqueeze(0)

    # Detach from any previous computation graph and enable gradients with respect to the input features
    # We only want dR/d(input) and dL/d(input)
    feat = feat.clone().detach().requires_grad_(True)

    # Forward pass through the inverse model
    # Output has shape [1, 2], corresponding to normalized [R, L]
    pred_norm = model(feat)

    # Separate normalized R and L predictions
    R_norm = pred_norm[0, 0]
    L_norm = pred_norm[0, 1]

    # Compute gradient of predicted R with respect to the input FRF features
    R_norm.backward(retain_graph=True)

    # Store the gradient vector dR/dfeat
    grad_R = feat.grad.detach().clone()[0]

    # Clear gradients before computing dL/dfeat
    feat.grad.zero_()

    # Compute gradient of predicted L with respect to the input FRF features
    L_norm.backward()

    # Store the gradient vector dL/dfeat
    grad_L = feat.grad.detach().clone()[0]

    # Number of frequency points.
    n_freq = len(freq_hz)

    # We reshape the gradients back to [n_freq, 2]
    # where the second dimension corresponds to the two acceleration components
    grad_R_real = grad_R[:2 * n_freq].reshape(n_freq, 2)
    grad_R_imag = grad_R[2 * n_freq:].reshape(n_freq, 2)

    grad_L_real = grad_L[:2 * n_freq].reshape(n_freq, 2)
    grad_L_imag = grad_L[2 * n_freq:].reshape(n_freq, 2)

    # For each frequency, this gives one scalar sensitivity value
    sens_R = grad_R_real.abs().sum(dim=1) + grad_R_imag.abs().sum(dim=1)
    sens_L = grad_L_real.abs().sum(dim=1) + grad_L_imag.abs().sum(dim=1)

    # Normalize each sensitivity curve by its maximum value
    # 1.0 = most important frequency for that parameter.
    sens_R = sens_R / sens_R.max().clamp_min(1e-12)
    sens_L = sens_L / sens_L.max().clamp_min(1e-12)

    # Return CPU tensors for plotting.
    return sens_R.cpu(), sens_L.cpu()

def compute_average_frequency_sensitivity(model, model_ctor, fixed_params, omega_grid, freq_hz, rl_pairs, norm_info, device="cpu"):

    # Lists that will store one sensitivity curve per R,L case
    all_sens_R = []
    all_sens_L = []

    # Label list of each pair
    labels = []

    # Loop through all selected R,L test cases
    for R_test, L_test in rl_pairs:

        # This generates the acceleration FRF that will be passed into the inverse model.
        sys = model_ctor(**fixed_params, R=R_test, L=L_test).to(device)

        # Compute the complex structural acceleration FRF for this case
        acc_test = sys.structural_accel_frf(omega_grid)

        # Compute local frequency sensitivity for this one FRF
        sens_R, sens_L = compute_frequency_sensitivity(model=model, acc_curve=acc_test, norm_info=norm_info, freq_hz=freq_hz)

        # Store the sensitivity curves
        all_sens_R.append(sens_R)
        all_sens_L.append(sens_L)
        labels.append(f"R={R_test}, L={L_test:.3f}")

    # Stack curves into matrices of shape
    # Each row corresponds to one R,L case
    all_sens_R_stack = torch.stack(all_sens_R, dim=0)
    all_sens_L_stack = torch.stack(all_sens_L, dim=0)

    # Average sensitivity across all cases.
    # This gives the frequency regions that are important on average
    mean_sens_R = all_sens_R_stack.mean(dim=0)
    mean_sens_L = all_sens_L_stack.mean(dim=0)

    # Standard deviation across cases
    # This shows how much the sensitivity varies from case to case
    std_sens_R = all_sens_R_stack.std(dim=0)
    std_sens_L = all_sens_L_stack.std(dim=0)

    # Normalize the mean curves for easier plotting.
    # Note: the standard deviations are not renormalized here after this operation
    mean_sens_R = mean_sens_R / mean_sens_R.max().clamp_min(1e-12)
    mean_sens_L = mean_sens_L / mean_sens_L.max().clamp_min(1e-12)

    return all_sens_R, all_sens_L, mean_sens_R, mean_sens_L, std_sens_R, std_sens_L, labels

@torch.no_grad()
def get_latent_features(model, X):

    model.eval()

    # Start from the input FRF feature vectors
    h = X

    # The result is the hidden 64-dimensional latent representation
    for layer in list(model.net.children())[:-2]:
        h = layer(h)

    return h

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_figure(fig, save_path=None, dpi=600):

    if save_path is None:
        return

    root, ext = os.path.splitext(save_path)
    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    if ext:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(root + ".pdf", bbox_inches="tight")
        fig.savefig(root + ".png", dpi=dpi, bbox_inches="tight")
        fig.savefig(root + ".svg", dpi=dpi, bbox_inches="tight")


def plot_sensitivity(freq_hz, all_sens, labels, save_path=None, title=None, ylabel="Normalized sensitivity", dpi=600, figsize=(5.8, 3.4)):
    freq_hz = _to_numpy(freq_hz)

    fig, ax = plt.subplots(figsize=figsize)

    for sens, label in zip(all_sens, labels):
        sens_np = _to_numpy(sens)

        ax.plot(freq_hz, sens_np, label=label, linewidth=1.1)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.tick_params(direction="in", top=True, right=True)

    # Put legend outside the plot
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=2.0, columnspacing=1.2)

    fig.tight_layout()
    _save_figure(fig, save_path=save_path, dpi=dpi)

    return fig, ax

def plot_sensitivity_heatmap(freq_hz, sens_mat, labels=None, save_path=None, title=None, colorbar_label="Normalized sensitivity", dpi=600, figsize=(5.8, 3.4), show_case_labels=False):

    freq_hz = _to_numpy(freq_hz)
    sens_mat = _to_numpy(sens_mat)

    n_cases = sens_mat.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(sens_mat, aspect="auto", origin="lower", extent=[freq_hz[0], freq_hz[-1], 1, n_cases], interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Blind case")

    ax.set_yticks(np.arange(1, n_cases + 1))

    if show_case_labels and labels is not None:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([str(i) for i in range(1, n_cases + 1)])

    if title is not None:
        ax.set_title(title)

    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    _save_figure(fig, save_path=save_path, dpi=dpi)

    return fig, ax

def plot_pca(Z_2d, c, label, save_path=None, title=None, dpi=600, figsize=(4.2, 3.6)):

    Z_2d = _to_numpy(Z_2d)
    c = _to_numpy(c)

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=c, s=14, linewidths=0.0)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")

    if title is not None:
        ax.set_title(title)

    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    _save_figure(fig, save_path=save_path, dpi=dpi)

    return fig, ax


def build_satmd_model(R, L, m2, kp, m1, c1, cp, k1, e_pz, C_p, force_amp, device):

    model = SATMDFrequencyGroundTruth(m1=m1, m2=m2, c1=c1, cp=cp, k1=k1, kp=kp, e_pz=e_pz, C_p=C_p, R=R, L=L, force_amp=force_amp).to(device)

    model.eval()
    return model


def select_frf_channel(Y, channel_idx=0):

    if Y.ndim == 1:
        return Y
    return Y[:, channel_idx]


def compute_global_frf_descriptor(R, L, m2, kp, nominal_model, omega_grid, fixed_params, channel_idx=0):

    device = omega_grid.device

    case_model = build_satmd_model(R=R, L=L, m2=m2, kp=kp, device=device, **fixed_params,)

    with torch.no_grad():
        Y_case = case_model.solve_frf(omega_grid)
        Y_nom = nominal_model.solve_frf(omega_grid)

    Y_case_j = select_frf_channel(Y_case, channel_idx=channel_idx)
    Y_nom_j = select_frf_channel(Y_nom, channel_idx=channel_idx)

    numerator = torch.linalg.norm(Y_case_j - Y_nom_j)
    denominator = torch.linalg.norm(Y_nom_j).clamp_min(1e-12)

    xi_frf = numerator / denominator

    return xi_frf.item()


def build_pysr_dataset(R_values,L_values,m2_values,kp_values,nominal_model,omega_grid,fixed_params,channel_idx=0,print_every=200):

    rows = []
    targets = []

    total_cases = (len(R_values) * len(L_values) * len(m2_values) * len(kp_values))

    counter = 0

    for R in R_values:
        for L in L_values:
            for m2 in m2_values:
                for kp in kp_values:

                    counter += 1

                    xi = compute_global_frf_descriptor(R=R, L=L, m2=m2, kp=kp, nominal_model=nominal_model, omega_grid=omega_grid, fixed_params=fixed_params, channel_idx=channel_idx)

                    rows.append([R, L, m2, kp])
                    targets.append(xi)

                    if counter % print_every == 0:
                        print(f"Computed {counter}/{total_cases} cases")

    X = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)

    return X, y


def normalize_inputs(X):

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0.0, 1.0, X_std)

    X_norm = (X - X_mean) / X_std

    return X_norm, X_mean, X_std


def make_pysr_model():

    model = PySRRegressor(
        niterations=800,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        model_selection="accuracy",
        maxsize=20,
        parsimony=0,
        variable_names=["R", "L", "m2", "kp"],
        random_state=42,
        parallelism="serial",
        deterministic=True,
        verbosity=1,
    )

    return model


def evaluate_symbolic_model(model_sr, X_train, X_test, y_train, y_test):

    y_pred_train = model_sr.predict(X_train)
    y_pred_test = model_sr.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
    }

    return y_pred_train, y_pred_test, metrics


def plot_predicted_vs_true(y_test, y_pred_test, save_path=None, dpi=600, figsize=(3.2, 2.8)):

    y_test = _to_numpy(y_test)
    y_pred_test = _to_numpy(y_pred_test)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_test, y_pred_test, alpha=0.7, s=16, linewidths=0.0)

    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.2, label="Ideal")

    ax.set_xlabel(r"True $\xi_{\mathrm{FRF}}$")
    ax.set_ylabel(r"Predicted $\hat{\xi}_{\mathrm{FRF}}$")

    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    _save_figure(fig, save_path=save_path, dpi=dpi)

    return fig, ax


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
    nominal_model = copy.deepcopy(model)
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

    save_path = "../../../logs/freq_response_comparison"
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
    # save_path = os.path.join(save_dir, "rl_inverse_net_5_256_5000.pth")
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

    results = evaluate_on_pairs(model=regressor,model_ctor=SATMDFrequencyGroundTruth,fixed_params=fixed_params,omega_grid=omega_t,eval_rl_pairs=eval_rl_pairs,norm_info=norm_info,device=device)

    for r in results:
        print(r)


    all_sens_R, all_sens_L, mean_sens_R, mean_sens_L, std_sens_R, std_sens_L, labels = compute_average_frequency_sensitivity(model=regressor, model_ctor=SATMDFrequencyGroundTruth, fixed_params=fixed_params, omega_grid=omega_t, freq_hz=freq_hz, rl_pairs=eval_rl_pairs, norm_info=norm_info, device=device)

    print(f"Mean sensitivity to R: mean = {mean_sens_R.mean().item():.6f}, std = {mean_sens_R.std().item():.6f}")
    print(f"Mean sensitivity to L: mean = {mean_sens_L.mean().item():.6f}, std = {mean_sens_L.std().item():.6f}")

    print(f"Across-case std for R sensitivity: mean = {std_sens_R.mean().item():.6f}, max = {std_sens_R.max().item():.6f}")
    print(f"Across-case std for L sensitivity: mean = {std_sens_L.mean().item():.6f}, max = {std_sens_L.max().item():.6f}")

    save_path_sensitivity_R = "../../../logs/sensitivity_to_R"
    title = None
    plot_sensitivity(freq_hz=freq_hz, all_sens = all_sens_R, labels = labels, save_path=save_path_sensitivity_R, title=title)


    save_path_sensitivity_L = "../../../logs/sensitivity_to_L"
    title = None
    plot_sensitivity(freq_hz=freq_hz, all_sens = all_sens_L, labels = labels, save_path=save_path_sensitivity_L, title=title)

    sens_R_mat = torch.stack(all_sens_R).numpy()
    sens_L_mat = torch.stack(all_sens_L).numpy()

    save_path_sensitivity_R_heatmap = "../../../logs/sensitivity_to_R_heatmap"
    title = None
    colorbar_label=r"$S_R(f)$"
    plot_sensitivity_heatmap(freq_hz=freq_hz, sens_mat=sens_R_mat, labels = labels, save_path=save_path_sensitivity_R_heatmap, title = title, colorbar_label=colorbar_label, show_case_labels=False,)


    save_path_sensitivity_L_heatmap = "../../../logs/sensitivity_to_L_heatmap"
    title = None
    colorbar_label=r"$S_L(f)$"
    plot_sensitivity_heatmap(freq_hz=freq_hz, sens_mat=sens_L_mat, labels = labels, save_path=save_path_sensitivity_L_heatmap, title = title, colorbar_label=colorbar_label, show_case_labels=False,)


    Z = get_latent_features(regressor, X)
    Z_2d = PCA(n_components=2).fit_transform(Z.cpu().numpy())

    label = "R [Ohm]"
    save_path_latent_R = "../../../logs/latent_R"
    plot_pca(Z_2d=Z_2d, c=y[:, 0], label=label, save_path=save_path_latent_R)

    label = "L [H]"
    save_path_latent_L = "../../../logs/latent_L"
    plot_pca(Z_2d=Z_2d, c=y[:, 1], label=label, save_path=save_path_latent_L)


    omega_grid = torch.tensor(omega, dtype=torch.float32, device=device)
    fixed_params = {
        "m1": m1,
        "c1": c1,
        "cp": cp,
        "k1": k1,
        "e_pz": e_pz,
        "C_p": C_p,
        "force_amp": force_amp
    }

    # Build nominal model
    nominal_model = build_satmd_model(R=R_nom, L=L_nom, m2=m2_nom, kp=kp_nom, device=device, **fixed_params)

    # Parameter ranges for PySR dataset
    R_values = np.linspace(8.0, 60.0, 10)
    L_values = np.linspace(32e-3, 300e-3, 10)
    m2_values = np.linspace(0.30, 0.60, 8)
    kp_values = np.linspace(50e3, 120e3, 8)


    # Build dataset
    X, y = build_pysr_dataset(
        R_values=R_values,
        L_values=L_values,
        m2_values=m2_values,
        kp_values=kp_values,
        nominal_model=nominal_model,
        omega_grid=omega_grid,
        fixed_params=fixed_params,
        channel_idx=0,
        print_every=200,
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Descriptor range:", y.min(), y.max())

    # Normalize inputs
    X_norm, X_mean, X_std = normalize_inputs(X)

    print("Input means [R, L, m2, kp]:")
    print(X_mean)

    print("Input stds [R, L, m2, kp]:")
    print(X_std)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    # Run PySR
    model_sr = make_pysr_model()
    model_sr.fit(X_train, y_train)

    # Evaluate
    y_pred_train, y_pred_test, metrics = evaluate_symbolic_model(model_sr=model_sr, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


    print("\n==============================")
    print("Best PySR expression")
    print("==============================")
    print(model_sr)

    print("\n==============================")
    print("Performance")
    print("==============================")
    print(f"Train MAE: {metrics['train_mae']:.6e}")
    print(f"Test  MAE: {metrics['test_mae']:.6e}")
    print(f"Train R2 : {metrics['train_r2']:.6f}")
    print(f"Test  R2 : {metrics['test_r2']:.6f}")

    # Plot result
    plot_predicted_vs_true(y_test, y_pred_test, save_path = "../../../logs/pred_vs_true")





