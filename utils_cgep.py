import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from typing import Callable




# ***************************************************************************
# --- complex activations ---
def monomial(z: np.ndarray, k: int) -> np.ndarray:
    assert isinstance(k, (int, np.integer)) and k >= 1
    return z ** k

def split_relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.real(z)) + 1j * np.maximum(0, np.imag(z))

def split_tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.real(z)) + 1j * np.tanh(np.imag(z))

def split_sign(z: np.ndarray) -> np.ndarray:
    return np.sign(np.real(z)) + 1j * np.sign(np.imag(z))

def split_modulus(z: np.ndarray) -> np.ndarray:
    return np.abs(np.real(z)) + 1j * np.abs(np.imag(z))

def z_relu(z: np.ndarray) -> np.ndarray:
    return np.where((np.real(z) >= 0.) & (np.imag(z) >= 0.), z, 0j)

def complex_cardioid(z: np.ndarray) -> np.ndarray:
    zb = np.abs(z)
    cosine = np.real(z) / (zb + 1e-10)
    return 0.5 * (1 + cosine) * z

def mod_relu(z: np.ndarray, b: float) -> np.ndarray:
    zb = np.abs(z)
    return np.maximum(zb + b, 0) * z / (zb + 1e-10)

def general_nonlinear(z: np.ndarray) -> np.ndarray:
    return (z + np.conj(z) ** 2) / (1 + np.abs(np.imag(z)))



activations = {
    'Monomial':         lambda k: (lambda z: monomial(z, k)),
    'Split-ReLU':       lambda : split_relu,
    'Split-Tanh':       lambda : split_tanh,
    'Split-Sign':       lambda : split_sign,
    'zReLU':            lambda : z_relu,
    'Complex Cardioid': lambda : complex_cardioid,
    'modReLU':          lambda b: (lambda z: mod_relu(z, b)),
    'Split-Modulus':    lambda : split_modulus,
    'Cosine':           lambda : np.cos,
    'Sine':             lambda : np.sin,
    'Modulus':          lambda : np.abs,
    r'$(z+\overline{z}^2)\,/\,(1+|\,\mathrm{Im}\,z\,|)$':   lambda : general_nonlinear
}
# ***************************************************************************



def clip_complex_array(zs: np.ndarray, cutoff: float) -> np.ndarray:

    assert cutoff > 0
    arr = zs.copy()
    abs_arr = np.abs(arr)
    mask = abs_arr >= cutoff
    arr[mask] *= cutoff / abs_arr[mask]

    return arr



def generate_complex_Gaussian_array(shape: tuple | list, v: float, tau: complex, 
                                    rng: np.random.Generator) -> np.ndarray:
    """
    Generate random complex Gaussian random variables 
    with E[z] = 0, E[|z|^2] = v, E[z^2] = tau. Must satisfy |tau| <= v.
    """
    shape = tuple(shape)
    v = float(v)
    tau = complex(tau)

    if v < 0.0:
        raise ValueError(f"v must be nonnegative, but v={v}.")

    abs_tau = abs(tau)

    if abs_tau > v:
        raise ValueError(
            f"Invalid parameters: |tau|={abs_tau} exceeds v={v}."
        )

    cov = 0.5 * np.array([
        [v + tau.real, tau.imag],
        [tau.imag,     v - tau.real],
    ])

    samples = rng.multivariate_normal(
        mean=np.zeros(2), cov=cov, size=shape,
        check_valid="raise", tol=1e-10,
    )

    return samples[..., 0] + 1j * samples[..., 1]



def estimate_gep_params(v_x: float, tau_x: complex, v_w: float, tau_w: complex, cutoff: float | None,
                        f: Callable, rng: np.random.Generator, n_samples: int = 10 ** 7):

    assert n_samples % 2 == 0
    assert v_x > np.abs(tau_x), f"Error: v_x ({v_x}) must be > |tau_x| ({np.abs(tau_x)})"
    assert v_w > np.abs(tau_w), f"Error: v_w ({v_w}) must be > |tau_w| ({np.abs(tau_w)})"
    assert (cutoff is None) or (cutoff is not None and cutoff > 0)

    v_h = v_x * v_w
    tau_h = tau_x * tau_w

    _zs = generate_complex_Gaussian_array(shape=(1000,), v=v_h, tau=tau_h, rng=np.random.default_rng(1))

    if np.max(np.abs(f(_zs) - f(- _zs))) < 1e-10:
        """Antithetic sampling is used only when f is an even function."""
        hs_half = generate_complex_Gaussian_array(shape=(n_samples // 2,), v=v_h, tau=tau_h, rng=rng)
        hs = np.concatenate((hs_half, - hs_half))
    else:
        hs = generate_complex_Gaussian_array(shape=(n_samples,), v=v_h, tau=tau_h, rng=rng)

    if cutoff is not None:

        hs = clip_complex_array(zs=hs, cutoff=cutoff)

    f_values = f(hs)
    mu = np.mean(f_values)
    kappa_1 = np.mean((f_values - mu) * np.conj(hs))
    kappa_2 = np.mean((f_values - mu) * hs)

    denom = v_h ** 2 - np.abs(tau_h) ** 2
    if denom < 1e-10:
        raise ValueError("v_h is too close to |tau_h|, causing singularity.")

    c_1 = (v_h * kappa_1 - np.conj(tau_h) * kappa_2) / denom
    c_2 = (v_h * kappa_2 - tau_h * kappa_1) / denom

    v_z = np.mean(np.abs(f_values - mu - c_1 * hs - c_2 * np.conj(hs)) ** 2)
    tau_z = np.mean((f_values - mu - c_1 * hs - c_2 * np.conj(hs)) ** 2)

    return {'c_1': c_1, 'c_2': c_2, 'v_z': v_z, 'tau_z': tau_z}



def single_SVD(params: dict, seed: int, mode: str, cutoff: float | None) -> tuple:

    assert mode in ['surrogate', 'nonlinear']
    
    if mode == 'surrogate':

        required_keys = {
            'n', 'd', 'p', 'v_x', 'v_w', 'tau_x', 'tau_w',
            'c_1', 'c_2', 'v_z', 'tau_z'
        }
    else:
        required_keys = {'n', 'd', 'p', 'v_x', 'v_w', 'tau_x', 'tau_w', 'f_tag', 'f_params'}

    missing_keys = required_keys - set(params.keys())
    assert not missing_keys, f"Missing keys in 'params': {missing_keys}"

    n, d, p = params['n'], params['d'], params['p']
    rng = np.random.default_rng(seed)
    X = generate_complex_Gaussian_array(shape=(n, d), v=params['v_x'],
                                        tau=params['tau_x'], rng=rng)
    W = generate_complex_Gaussian_array(shape=(d, p), v=params['v_w'],
                                        tau=params['tau_w'], rng=rng)
    H = (X @ W) / np.sqrt(d)

    if cutoff is not None:

        H = clip_complex_array(zs=H, cutoff=cutoff)

    if mode == 'surrogate':

        if params['v_z'] > 1e-10:
            Z = generate_complex_Gaussian_array(shape=(n, p), v=params['v_z'], tau=params['tau_z'], rng=rng)
        else:
            Z = np.zeros((n, p), dtype=complex)
            
        Q = params['c_1'] * H + params['c_2'] * np.conj(H) + Z

    else:
        f = activations[params['f_tag']](**params['f_params'])
        Q = f(H)

    assert Q.shape == (n, p)
    singvals = np.linalg.svd(Q / np.sqrt(n), compute_uv=False)
    rank_Q = np.count_nonzero(singvals > 1e-10)
    nonzero_singvals = singvals[:rank_Q]
    nonzero_eigvals = nonzero_singvals ** 2

    return nonzero_eigvals, nonzero_singvals



def cc(z: complex, precision: int | tuple = 3) -> str:
    
    """Format a complex number for use in a Matplotlib math label."""

    imaginary_unit = r"\,i"
    
    if isinstance(precision, tuple):
        assert len(precision) == 2
        precision_g, precision_f = precision
    elif isinstance(precision, int):
        precision_g, precision_f = precision, precision
    else:
        raise NotImplementedError

    def format_component(x: float) -> str:

        zero_threshold = 0.5 * 10.0 ** (- precision_f)

        if abs(x) < zero_threshold:
            return "0"

        if abs(x) < 1.0:
            text = f"{x:.{precision_f}f}"
        else:
            text = f"{x:.{precision_g}g}"

        if "e" not in text.lower() and "." in text:
            text = text.rstrip("0").rstrip(".")

        return "0" if text in {"-0", "+0"} else text

    real_str = format_component(float(np.real(z)))
    imag_value = float(np.imag(z))
    imag_str = format_component(abs(imag_value))

    real_is_zero = real_str == "0"
    imag_is_zero = imag_str == "0"

    if imag_is_zero:
        return real_str

    if imag_str == "1":
        imag_str = imaginary_unit
    else:
        imag_str = f"{imag_str}{imaginary_unit}"

    if real_is_zero:
        return imag_str if imag_value > 0.0 else f"-{imag_str}"

    sign = "+" if imag_value > 0.0 else "-"

    return f"{real_str}{sign}{imag_str}"



def plot_distribution(data_Q: np.ndarray, data_f: np.ndarray, params: dict,
                      precision: int | tuple, n_bins: int,
                      ext: str | int | None = None,
                      legend_anchor: tuple | None = None,
                      x_range: tuple | None = None, y_range: tuple | None = None,
                      return_file_type: str | None = None) -> None:

    required_keys = {'n', 'd', 'p', 'n_mc', 'f_tag', 'f_params'}
    missing_keys = required_keys - params.keys()
    assert not missing_keys, f"Missing keys in params: {sorted(missing_keys)}"
    
    if return_file_type is None:
        assert ext is None
    else:
        assert return_file_type in ['pdf', 'png']
        assert ext is not None

    n, d, p = params['n'], params['d'], params['p']

    f_params = params['f_params']
    if f_params:
        arg = "(" + ", ".join(
            f"${key}={value}$"
            for key, value in f_params.items()
        ) + ")"

        arg_2 = "(" + "_".join(
            f"{key}_{value}"
            for key, value in f_params.items()
        ) + ")"
    else:
        arg = ''
        arg_2 = ''
    
    sep = r'\hspace{0.2}'
    
    title = (
        (
            rf'$f$ : {params['f_tag']}' if not params['f_params']
            else (rf'$f$ : {params['f_tag']} ' + arg)
        )
        + r'$,$' + '\n' + rf'$(n,d,p)=({n},{sep}{d},{sep}{p}),{sep}{sep}$' \
        + rf'$(c_1,c_2)=({cc(params['c_1'], precision=precision)},{sep}{cc(params['c_2'], precision=precision)}),$'
        + '\n' + rf'$(v_w,\tau_w,\tau_x)=({cc(params['v_w'])},{sep}{cc(params['tau_w'])},{sep}{cc(params['tau_x'])})$'
        + rf'$,{sep}{sep}N_{{\mathrm{{MC}}}}={params['n_mc']}$'
    )

    plt.figure(figsize=(7.5, 5))

    def decide_max_plot_range(data: np.ndarray) -> float:

        q1, q3 = np.quantile(data, [0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        is_outlier = data > upper
        inliers = data[~is_outlier]
        d_max_plot = 1.08 * np.max(inliers)

        return d_max_plot

    d_max_plot_f = decide_max_plot_range(data=data_f)
    d_max_plot_Q = decide_max_plot_range(data=data_Q)
    d_max_plot = max(d_max_plot_Q, d_max_plot_f)

    def hist_plotter(data: np.ndarray, color_name: str, drawstyle: str,
                     plot_label: str, d_max_plot: float):

        q1, q3 = np.quantile(data, [0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        is_outlier = data > upper
        inliers = data[~is_outlier]
        print(f'Portion of outliers : {1.0 - len(inliers) / len(data):.6g}. ' + \
              ('' if np.sum(is_outlier) == 0 else f'Max outlier : {np.max(data[is_outlier]):.6g}.'))
        n_bins_ = n_bins if ('steps' in drawstyle) else int(1.5 * n_bins)
        bins = np.linspace(0.0, d_max_plot, n_bins_ + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(inliers, bins=bins, density=False)
        bin_widths = np.diff(bins)
        hist_vals = counts / (len(data) * bin_widths)

        if 'steps' in drawstyle:
            plt.fill_between(bin_centers, 0, hist_vals, step='mid', alpha=0.25, color=color_name)
        
        plt.plot(bin_centers, hist_vals, color=color_name, linewidth=1.75, 
                 drawstyle=drawstyle, label=plot_label)

    hist_plotter(data=data_f, color_name='tab:red', drawstyle='steps-mid',
                 plot_label='Original model', d_max_plot=d_max_plot)
    
    hist_plotter(data=data_Q, color_name='royalblue', drawstyle='default',
                 plot_label='Surrogate model', d_max_plot=d_max_plot)
    
    # Formatting
    if x_range is None:
        plt.xlim(0.0, d_max_plot)
    else:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    
    plt.title(title)
    plt.xlabel('Singular value')
    plt.ylabel('Density')
    plt.grid(color='gray', linestyle=':')
    if legend_anchor is None:
        plt.legend(loc='best', framealpha=0.9)
    else:
        plt.legend(framealpha=0.9, bbox_to_anchor=legend_anchor)

    if return_file_type is not None:

        try:
            f_tag_str = params['f_tag'].replace(' ', '_')
            filename = f'fg_{f_tag_str}{arg_2}_{ext}.' + return_file_type
            plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.05)
        except:
            filename = f'fg_TEMP_{ext}.' + return_file_type
            plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.05)

    plt.show()



class Runner:

    def __init__(self, params: dict, cutoff: float | None = None):

        required_keys = {'n', 'd', 'p', 'v_x', 'v_w', 'tau_x', 'tau_w', 'f_tag', 'f_params'}
        missing_keys = required_keys - params.keys()
        assert not missing_keys, f"Missing keys in params: {sorted(missing_keys)}"

        self.cutoff = cutoff
        f = activations[params['f_tag']](**params['f_params'])
        cgep_params = estimate_gep_params(v_x=params['v_x'], tau_x=params['tau_x'], v_w=params['v_w'],
                                          tau_w=params['tau_w'], f=f, cutoff=cutoff, rng=np.random.default_rng(0))
        self.params = params | cgep_params

    def run(self, n_mc: int, n_jobs: int, seed: int):

        self.params = self.params | {'n_mc': n_mc}

        # Surrogate Monte Carlo simulation------------------------------
        results_Q = Parallel(n_jobs=n_jobs)(
            delayed(single_SVD)(self.params, seed + i, 'surrogate', self.cutoff) for i in range(1, 1 + n_mc)
        )
        _, singvals_Q = zip(*results_Q)
        self.data_Q = np.concatenate(singvals_Q)
        #---------------------------------------------------------------

        # Original Monte Carlo simulation-------------------------------
        results_f = Parallel(n_jobs=n_jobs)(
            delayed(single_SVD)(self.params, seed + i, 'nonlinear', self.cutoff) for i in range(1, 1 + n_mc)
        )
        _, singvals_f = zip(*results_f)
        self.data_f = np.concatenate(singvals_f)
        #---------------------------------------------------------------

    def generate_plot(self, n_bins: int, return_file_type: str | None, ext: str | int | None,
                      precision: int | tuple = 3, legend_anchor: tuple | None = None,
                      x_range: tuple | None = None, y_range: tuple | None = None):
        
        plot_distribution(data_Q=self.data_Q, data_f=self.data_f, params=self.params, n_bins=n_bins, legend_anchor=legend_anchor,
                          x_range=x_range, y_range=y_range, return_file_type=return_file_type, ext=ext, precision=precision)



