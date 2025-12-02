# ETCalcPyMOL, v0.1
import sys, csv, itertools

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from math import factorial
from scipy.optimize import minimize, curve_fit, brentq
from scipy.integrate import odeint
import networkx as nx

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

import tkinter as tk
from tkinter import ttk, filedialog
import tkinter.font as tkfont

import MDAnalysis as mda
from pymol import cmd, cgo

# ---------------------- constants (eV-based) ----------------------
kB_eV_per_K = 8.617333262145e-5   # Boltzmann constant [eV/K]
hbar_eV_s   = 6.582119569e-16     # Planck constant / 2π [eV·s]
_COULOMB_CONST = 14.3996454784255 # e^2/(4 π ε0) in eV·Å

# Default Marcus parameters
DEF_LAMBDA = 0.70
DEF_DELTAG = -0.50
DEF_TK     = 298.15
DEF_H0     = 0.050
DEF_BETA   = 1.10
DEF_R0     = 3.50
DEF_DeltaB = 1.00
DEF_Floor  = 0.00
DEF_POW    = 1.00

# Donor and acceptor atom names
donacc_names_dict={"TRP": {"CG","CD1","NE1","CE2","CD2","CZ2","CH2","CE3","CZ3"},
                   "TYR": {"CG","CD1","CE1","CZ","CD2","CE2","OH"},
                   "PHE": {"CG","CD1","CE1","CZ","CD2","CE2"},
                   "HIS": {"CG","ND1","CD2","CE1","NE2"},
                   "FADdefault": {"N1","C2","N3","C4","C4A","C4X","N5","C5A","C6","C7","C8","N10","C9","C9A"},
                   "FADselec": {"N5","C4A","C4X","C5A","C5X"},
                   "AMP": {"N1","C2","N3","C4","C5","C6","N7","C8","C9","N1A","C2A","N3A","C4A","C5A","C6A","N7A","C8A","C9A"},
                   "SF4": {"FE1","FE2","FE3","FE4","S1","S2","S3","S4"},
                   "BACKBONE": {"N","CA","C","O","OXT"}
}

namesFAD= ["FAD","FADH","FADH2","FDA","FMN","FMNH","FMNH2"]
namesAMP= ["AMP","ADP","ATP","CMP","ACK","ADE","ADN","NAI","NAD","NAP","NDP"]
namesArom=['TRP', 'TYR', 'PHE', 'HIS'] + namesFAD + namesAMP

# --- some usage texts, need updating---
USAGE_RATE = """
Usage:
  et_rate_marcus [geom_mode] obj, don_chain, don_resi, acc_chain, acc_resi
                      [, lambda_eV[, deltaG_eV[, T_K[, H0[, beta[, R0[, orient_mode[, orient_floor[, orient_pow]]]]]]]]]

Examples:
  et_rate_marcus default, 1DNP, A, 382, A, 472
  et_rate_marcus default, 1DNP, A, 382, A, 472, 0.70, -0.50, 298.15, 0.070, 1.00, 3.50, on, 0.1, 1.0

Defaults:
  lambda_eV=0.70 eV, deltaG_eV=-0.50 eV, T_K=298.15 K, H0=0.070 eV,
  beta=1.00 Å^-1, R0=3.50 Å, orient_mode='off', orient_floor=0.0, orient_pow=1.0

Description:
  Computes Marcus ET rate using distance-based electronic coupling H_ab.
  Updates arrow/distance visualization in PyMOL.
"""

USAGE_MLJ = """
Usage:
  et_rate_MLJ [geom_mode] obj, don_chain, don_resi, acc_chain, acc_resi
              [, lambda_s_eV[, deltaG_eV[, T_K[, H0[, beta[, R0[, S_csv[, hw_csv[, vmax_csv[, orient_mode[, orient_floor[, orient_pow]]]]]]]]]]]]]

Description:
  Marcus–Levich–Jortner ET rate with single or multi-vibrational modes.
  CSV strings for S, hw, and vmax must be equal length in multi-mode case.
"""

# ----------------------Abstract Theory Engine ----------------------
#
class ETEngine(ABC):
    """
    Abstract base class for Electron Transfer rate engines.
    Subclasses implement specific theories (e.g., Marcus, MLJ, Redfield).
    
    Required: compute_rate method for core rate calculation.
    Optional: fit_params for calibration, compute_reorg for dynamic lambda.
    
    Usage: Instantiate with params, then call engine.compute_rate(Hab, other_params).
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        params: Dictionary of theory-specific parameters (e.g., {'lambda_eV': 0.7, 'deltaG_eV': -0.5}).
        """
        self.params = params
    
    @abstractmethod
    def compute_rate(self, Hab_eV: float, geom_params: Dict[str, Any] = None) -> float:
        """
        Compute ET rate k [s^-1] for given electronic coupling Hab_eV.
        
        Args:
            Hab_eV: Electronic coupling [eV].
            geom_params: Optional dict with geometry-dependent params (e.g., {'R_eff': 5.0}).
        
        Returns:
            k [s^-1].
        """
        pass
    
    def fit_to_target(self, k_target: float, geom_params: Dict[str, Any],
                      fit_keys: List[str], bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Fit engine params to match target rate using scipy.optimize.
        
        Args:
            k_target: Target rate [s^-1].
            geom_params: Geometry dict (e.g., for Hab calculation).
            fit_keys: List of param keys to fit (e.g., ['H0', 'lambda_eV']).
            bounds: Optional bounds dict.
        
        Returns:
            Fitted params dict.
        """
        from scipy.optimize import minimize
        
        def objective(x):
            # Update params with trial values
            trial_params = self.params.copy()
            for i, key in enumerate(fit_keys):
                trial_params[key] = x[i]
            # Temp engine with trial params
            trial_engine = self.__class__(trial_params)
            # Compute Hab if needed (e.g., from geom)
            if 'R_eff' in geom_params:
                H0 = trial_params.get('H0', 0.05)
                beta = trial_params.get('beta', 1.1)
                R0 = trial_params.get('R0', 3.5)
                Hab = H0 * np.exp(-beta * (geom_params['R_eff'] - R0))
            else:
                Hab = geom_params.get('Hab_eV', 0.01)
            k_calc = trial_engine.compute_rate(Hab)
            return abs(np.log10(k_calc) - np.log10(k_target))
        
        x0 = [self.params[key] for key in fit_keys]
        bnds = [(bounds[key][0], bounds[key][1]) for key in fit_keys] if bounds else None
        res = minimize(objective, x0, bounds=bnds)
        if res.success:
            fitted = {key: val for key, val in zip(fit_keys, res.x)}
            self.params.update(fitted)
            return fitted
        else:
            raise ValueError(f"Fit failed: {res.message}")
    
    def compute_reorg_from_traj(self, traj_energies: List[float], T_K: float) -> float:
        """
        Example hook for dynamic lambda from energy gaps (Kuznetsov-Ulstrup).
        Override in subclasses for theory-specific implementations.
        
        Args:
            traj_energies: List of equilibrium/vertical energy differences from MD.
            T_K: Temperature [K].
        
        Returns:
            lambda_eV [eV].
        """
        # Placeholder: Mean squared fluctuation
        delta_E = np.array(traj_energies)
        lambda_eV = np.var(delta_E) / (2 * kB_eV_per_K * T_K)
        self.params['lambda_eV'] = lambda_eV
        return lambda_eV

# ---------------------- Theory Implementation classes ----------------------

class MarcusEngine(ETEngine):
    """Classical Marcus Theory Engine."""
    
    def compute_rate(self, Hab_eV: float, geom_params: Dict[str, Any] = None) -> float:
        lambda_eV = self.params['lambda_eV']
        deltaG_eV = self.params['deltaG_eV']
        T_K = self.params['T_K']
        # Reuse existing _marcus_rate
        return _marcus_rate(Hab_eV, lambda_eV, deltaG_eV, T_K)

class MLJEngine(ETEngine):
    """Marcus-Levich-Jortner Engine for vibronic effects."""
    
    def compute_rate(self, Hab_eV: float, geom_params: Dict[str, Any] = None) -> float:
        lambda_s_eV = self.params['lambda_s_eV']
        deltaG_eV = self.params['deltaG_eV']
        T_K = self.params['T_K']
        S_list = self.params.get('S_list', [1.0])
        hw_list = self.params.get('hw_list', [0.18])
        vmax_list = self.params.get('vmax_list', [10])
        # Reuse existing _mlj_rate_multi
        return _mlj_rate_multi(Hab_eV, lambda_s_eV, deltaG_eV, T_K, S_list, hw_list, vmax_list)

class RedfieldEngine(ETEngine):
    """
    Fixed Redfield Theory Engine: Secular approximation for ET rate.
    k = 2 * (Hab / ħ)^2 * γ / [(Δ / ħ)^2 + γ^2 ]   [s^-1]
    where γ = 2 λ kT / ħ (classical dephasing from reorganization).
    Recovers weak-coupling limit; depends on Hab, ΔG, λ, T.
    
    Params:
        - 'deltaG_eV': Driving force [eV].
        - 'lambda_eV': Reorg energy for γ_deph [eV] (auto-computes γ if not set).
        - 'T_K': Temperature [K].
        - 'Hab_eV': For direct; for bridges, pass as effective.
        - For multi-site: Use 'site_energies_eV' list, 'edge_couplings_eV' list;
          approximates k via product or sum, but for simplicity, uses overall ΔG.
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.hbar = hbar_eV_s
        self.kB = kB_eV_per_K
        
        # Auto-set gamma_deph from lambda if provided
        if 'lambda_eV' in self.params and 'gamma_deph_s1' not in self.params:
            lam = self.params['lambda_eV']
            T = self.params['T_K']
            gamma_deph = 2 * lam * self.kB * T / self.hbar  # [s^-1]
            self.params['gamma_deph_s1'] = gamma_deph
        self.params.setdefault('gamma_deph_s1', 5e13)  # Default ~ for λ=0.7 eV, 298K
        self.params.setdefault('N_sites', 2)
    
    def compute_rate(self, Hab_eV: float, geom_params: Dict[str, Any] = None) -> float:
        """Analytical secular Redfield rate for D->A transfer."""
        deltaG_eV = self.params['deltaG_eV']
        gamma = self.params['gamma_deph_s1']
        
        # To angular freq [rad/s]
        Hab = Hab_eV / self.hbar
        delta = deltaG_eV / self.hbar
        
        # Lorentzian lineshape
        denom = delta**2 + gamma**2
        k = 2 * Hab**2 * gamma / denom if denom != 0 else 0.0
        
        # For multi-site: Placeholder - use effective Hab_eff from geom_params if provided
        if self.params['N_sites'] > 2 and 'Hab_eff_eV' in (geom_params or {}):
            Hab = (geom_params['Hab_eff_eV'] / self.hbar)
            k = 2 * Hab**2 * gamma / denom  # Approximate as effective 2-site
        
        return k
    
    def fit_to_target(self, k_target: float, geom_params: Dict[str, Any],
                      fit_keys: List[str], bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """Standard fit, now with correct dependence."""
        return super().fit_to_target(k_target, geom_params, fit_keys, bounds)


class TSHException(Exception):
    """Custom exception for TSH failures."""
    pass

class TSH_Engine(ETEngine):
    """
    Non-Adiabatic Trajectory Surface Hopping (TSH) Engine with Rescaling + Rejection.
    Semiclassical: Classical nuclear trajectories with stochastic hops based on NACs.
    For ET: 2-state model (D/A), with diabatic energies E_D(t)=0, E_A(t)=ΔG + λ (Q - Q_opt)^2 / 2
    (harmonic for reorg). Hop prob dP = 2 Re[ρ_DA * NAC] dt, fewest-switches.
    
    Frustration: Rescale momentum for successful hops (K' >= 0); reject and continue on current surface if K' < 0.
    Statistics: Tracks total hop attempts and frustrated rejections; reports frustration_rate.
    
    Params:
        - 'deltaG_eV': Driving force [eV].
        - 'lambda_eV': Reorg energy [eV].
        - 'T_K': Temperature [K] (for initial Q, P from Boltzmann).
        - 'mass_amu': Reduced mass [amu] (default 100 for protein mode).
        - 'omega_cm': Freq [cm^-1] for harmonic bath (default 1000).
        - 'n_trajectories': Number of runs (default 100).
        - 't_max_ps': Max time [ps] (default 10).
        - 'dt_fs': Timestep [fs] (default 0.1 for better coherence resolution).
    
    compute_rate: Averages survival prob fit to 1 - exp(-k t) over trajectories.
    Returns: k [s^-1] (positive; fallback to initial slope if needed).
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.hbar = hbar_eV_s
        self.kB = kB_eV_per_K
        self.mass_amu = params.get('mass_amu', 100.0)  # amu
        self.mass_kg = self.mass_amu * 1.660539e-27  # kg
        self.omega_cm = params.get('omega_cm', 1000.0)  # cm^-1
        self.omega_rad_s = self.omega_cm * 100 * 2 * np.pi * 3e8 / 100  # rad/s (c in cm/s)
        self.omega_eV_s = self.omega_rad_s * self.hbar  # for energy
        self.n_traj = params.get('n_trajectories', 100)
        self.t_max_s = params.get('t_max_ps', 10.0) * 1e-12
        self.dt_s = params.get('dt_fs', 0.1) * 1e-15  # Finer default
        self.times_s = np.arange(0, self.t_max_s, self.dt_s)

    def _diabatic_energies_better(self, t: float): ### not implemented yet ...
        """
        Returns the N × N diabatic energy matrix E(t) in cm⁻¹ at time t (fs)
        Correct physics:
          • Diagonal = site energies + bath modulation + λ (reorg) shift
          • Off-diagonal = static electronic couplings J_ij (no time-dep unless specified)
          • Zero of energy = ground-state chromophore (not excited-state minimum)
        """
        N = self.n_sites
        E = np.zeros((N, N), dtype=np.float64)

        # 1. Static site excitation energies (vertical Franck-Condon, in cm⁻¹)
        E0 = np.asarray(self.site_energies)  # shape (N,)

        # 2. Reorganization energy shift (Stokes shift / 2)
        #    → each excited-state minimum is lowered by λ
        lambda_reorg = np.asarray(self.reorganization_energies)  # (N,) or scalar

        # 3. Time-dependent bath modulation (independent OU processes per site)
        if self.bath is not None:
            deltaE_bath = self.bath.energy_fluctuations(t)  # shape (N,)
        else:
            deltaE_bath = np.zeros(N)

        # 4. Build diagonal (diabatic site energies at time t)
        np.fill_diagonal(E, E0 + lambda_reorg + deltaE_bath)
    
        # 5. Off-diagonal: electronic couplings (permanent, in cm⁻¹)
        #    self.couplings is upper-triangle or full symmetric matrix
        couplings = np.asarray(self.couplings)
        if couplings.ndim == 1:  # vectorized upper-triangle format
            idx_i, idx_j = np.triu_indices(N, k=1)
            E[idx_i, idx_j] = couplings
            E[idx_j, idx_i] = couplings
        else:  # full matrix already
            E[:] += couplings
            E[:] += couplings.T
            np.fill_diagonal(E, np.diag(E))  # ensure diagonal untouched
        return E  # cm⁻¹

    def _adiabatic_energies(self, t): ### not implemented yet ...
        H = self._diabatic_energies(t)
        return np.linalg.eigh(H)[0]  # sorted eigenvalues in cm⁻¹
    
    def _diabatic_energies(self, Q, P):
        """E_D(Q,P)=0 + K, E_A(Q,P)=ΔG + λ (Q - Q_opt)^2 / 2 + K [eV]. K shared."""
        deltaG = self.params['deltaG_eV']
        lam = self.params['lambda_eV']
        Q_opt = -deltaG / (2 * lam)  # Optimum crossing
        V_A = deltaG + 0.5 * lam * (Q - Q_opt)**2
        K = 0.5 * P**2 / (self.mass_kg * self.omega_rad_s**2) * (self.omega_eV_s / self.hbar)  # eV equiv
        return np.array([0.0 + K, V_A + K])  # Both include K
    
    def _nonadiabatic_coupling(self, Q, P, Hab):
        """NAC_{DA} = <D| d/dQ |A> ≈ Hab / (E_D - E_A) [1/Å]."""
        E = self._diabatic_energies(Q, P)
        denom = E[0] - E[1]
        return Hab / denom if abs(denom) > 1e-6 else 0.0
    
    def _rk4_step_rho(self, rho, Hab, V_diff, dt):
        """RK4 integration for rho (vectorized for stability)."""
        k1 = -1j * np.array([[0, Hab], [Hab, V_diff]]) @ rho / self.hbar
        k2 = -1j * np.array([[0, Hab], [Hab, V_diff]]) @ (rho + 0.5 * dt * k1) / self.hbar
        k3 = -1j * np.array([[0, Hab], [Hab, V_diff]]) @ (rho + 0.5 * dt * k2) / self.hbar
        k4 = -1j * np.array([[0, Hab], [Hab, V_diff]]) @ (rho + dt * k3) / self.hbar
        return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _propagate_rho(self, rho, Q, P, active, Hab):
        """Semiclassical: Propagate coeffs via RK4 TDSE, velocities via force on active state."""
        # V_diff without K (cancels in H)
        V_diff = self._diabatic_energies(Q, 0)[1] - self._diabatic_energies(Q, 0)[0]  # ΔV
        rho_new = self._rk4_step_rho(rho, Hab, V_diff, self.dt_s)
        # Velocities: dQ/dt = P/m, dP/dt = -dV/dQ on active state
        m = self.mass_kg
        dQdt = P / m
        if active == 0:  # D: flat, force=0
            dPdt = 0.0
        else:  # A: harmonic
            Q_opt = -self.params['deltaG_eV'] / (2 * self.params['lambda_eV'])
            dPdt = -self.params['lambda_eV'] * (Q - Q_opt)  # -dV_A/dQ
        return rho_new - rho, np.array([dQdt, dPdt])  # Return delta_rho for update
    
    def _fewest_switches_hop(self, rho, active_state, NAC, dt, Q, P, stats, Hab):
        """Hop prob dP = 2 Re[ρ_{12} * NAC * v] dt. Rescaling + rejection for frustration."""
        v = P / self.mass_kg  # Velocity
        if active_state == 0:  # D active, prob to A
            g = 2 * np.real(rho[0,1] * np.conj(NAC)) * v * dt
        else:  # A active, prob to D
            g = 2 * np.real(rho[1,0] * np.conj(NAC)) * v * dt
        
        attempt = False
        if np.random.rand() < max(g, 0):
            attempt = True
            stats['total_attempts'] += 1
            # Attempt hop
            new_state = 1 - active_state
            E_old = self._diabatic_energies(Q, P)[active_state]
            V_new = self._diabatic_energies(Q, 0)[new_state]  # V without K
            V_old = self._diabatic_energies(Q, 0)[active_state]
            K_old = E_old - V_old
            delta_V = V_new - V_old
            K_new = K_old - delta_V  # Total E conserved: K' = K - ΔV
            
            if K_new >= 0:
                # Successful hop: Rescale P to conserve E (direction preserved)
                P_new = np.sign(P) * np.sqrt(2 * self.mass_kg * K_new / (self.omega_eV_s / self.hbar) * self.omega_rad_s**2)
                P = P_new
                active_state = new_state
            else:
                # Frustrated hop: Reject, increment frustration
                stats['frustrated_hops'] += 1
                # Continue on current surface (no change to P/Q)
        
        return active_state, P  # Return updated P (unchanged if rejected)
    
    def _run_single_trajectory(self, initial_Q, initial_P, Hab):
        """Run one TSH trajectory with local stats."""
        Q, P = initial_Q, initial_P
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # Start in D (state 0)
        active = 0
        pops = np.zeros(len(self.times_s))
        local_stats = {'total_attempts': 0, 'frustrated_hops': 0}  # Per traj
        
        for i, t in enumerate(self.times_s):
            pops[i] = np.abs(rho[1,1])**2  # Pop A
            NAC = self._nonadiabatic_coupling(Q, P, Hab)
            delta_rho, dstate = self._propagate_rho(rho, Q, P, active, Hab)
            rho += delta_rho
            rho = rho / np.linalg.norm(rho)  # Normalize
            active, P = self._fewest_switches_hop(rho, active, NAC, self.dt_s, Q, P, local_stats, Hab)
            # Update Q, P (Euler)
            dQ, dP = dstate
            Q += dQ * self.dt_s
            P += dP * self.dt_s
        
        return pops, local_stats
    
    def compute_rate(self, Hab_eV: float, geom_params: Dict[str, Any] = None, debug=False) -> float:
        """Average over trajectories, fit to extract k. Returns k [s^-1] (positive)."""
        from scipy.optimize import curve_fit
        def exp_decay(t, k): return 1 - np.exp(-k * t)
        
        Hab = Hab_eV or self.params.get('Hab_eV', 0.022)
        avg_pop_A = np.zeros(len(self.times_s))
        total_stats = {'total_attempts': 0, 'frustrated_hops': 0}
        
        # Initial conditions: Boltzmann for Q, P at T
        beta = 1 / (self.kB * self.params['T_K'])
        sigma_Q = np.sqrt(self.kB * self.params['T_K'] / self.params['lambda_eV'])  # Harmonic approx
        for traj_i in range(self.n_traj):
            initial_Q = np.random.normal(0, sigma_Q)
            initial_P = np.random.normal(0, np.sqrt(self.mass_kg * self.kB * self.params['T_K']))
            traj_pop, traj_stats = self._run_single_trajectory(initial_Q, initial_P, Hab)
            avg_pop_A += traj_pop / self.n_traj
            # Aggregate stats
            for key in total_stats:
                total_stats[key] += traj_stats[key]
            if debug and traj_i < 5:  # Debug first few traj
                print(f"Traj {traj_i}: max pop_A = {np.max(traj_pop):.3f}, attempts = {traj_stats['total_attempts']}")
        
        # Compute overall frustration rate
        total_stats['frustration_rate'] = total_stats['frustrated_hops'] / max(total_stats['total_attempts'], 1)
        
        # Robust fit: Initial slope fallback, bound k>0
        initial_slope = (avg_pop_A[1] - avg_pop_A[0]) / self.dt_s if len(avg_pop_A) > 1 else 0
        try:
            popt, _ = curve_fit(exp_decay, self.times_s, avg_pop_A, p0=[max(initial_slope, 1e10)], bounds=([0], [np.inf]))
            k = max(popt[0], 0)
        except:
            k = max(initial_slope, 0)  # Positive fallback
        
        # Print stats
        print(f"TSH Stats: {total_stats['total_attempts']} attempts, {total_stats['frustrated_hops']} frustrated ({total_stats['frustration_rate']*100:.1f}% rate)")
        
        return k
    
    def fit_to_target(self, k_target: float, geom_params: Dict[str, Any],
                      fit_keys: List[str], bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """Standard fit; TSH stochasticity may need more traj for convergence."""
        return super().fit_to_target(k_target, geom_params, fit_keys, bounds)

class EmpiricalMarkusEngine(ETEngine):
    """
    Empirical Moser–Dutton/Marcus hybrid ET rate engine.

    Formula:
        log10(k) = C0 - beta/2.303 * (R_eff-R0) + Marcus-like ΔG correction

    Params:
        C0         : intercept in log10(k) scale (default 13.0)
        beta       : distance decay constant [Å^-1] (default 1.1)
        lambda_eV  : reorganization energy [eV]
        deltaG_eV  : driving force [eV]
        T_K        : temperature [K]
        R0         : 3.6 Angstrom originally, 3.0 Angstrom Zhong and coworkers 
    """
    def compute_rate(self, Hab_eV=None, geom_params=None):
        C0   = self.params.get('C0', 13.0)
        beta = self.params.get('beta', 1.1)
        lambda_eV = self.params['lambda_eV']
        deltaG_eV = self.params['deltaG_eV']
        T_K  = self.params['T_K']
        R0   = self.params['R0']
        R_eff = geom_params.get('R_eff') if geom_params else None
        if R_eff is None:
            raise ValueError("EmpiricalMarkusEngine requires 'R_eff' in geom_params")

        log10_k = 13 - beta/2.303 * (R_eff-R0) - 3.1 * ((deltaG_eV + lambda_eV)**2) / lambda_eV
        return 10.0 ** log10_k

_engine_map = {
    'marcus': MarcusEngine,
    'mlj': MLJEngine,
    'redfield': RedfieldEngine,
    'tsh': TSH_Engine,
    'empirical_markus': EmpiricalMarkusEngine
}

# ---------------------- Example Usage Integration ----------------------
# In et_rate_marcus (or new command et_rate_redfield):
# params = {'deltaG_eV': dG, 'T_K': T, 'Hab_eV': Hab, 'gamma_relax_s1': 1e12}
# engine = RedfieldEngine(params)
# k = engine.compute_rate()
# For multi-bridge: In bridge functions, build chain with N = len(bridges)+2, site_energies from Emode,
# edge_couplings from Vedges, then k = engine.compute_rate()

# --- utilities ---

def _normalize_np(v):
    nrm = np.linalg.norm(v)
    return None if nrm <= 1e-12 else v / nrm

def _autocorr_fft(x):
    """Fast autocorrelation via FFT, normalised."""
    x = np.array(x, dtype=float)
    n = len(x)
    x -= np.mean(x)
    f = np.fft.fft(x, n*2)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf

def _exp_decay(t, tau_c):
    return np.exp(-t / tau_c)

def _complex_solve(M, b):
    # Solve M x = b for complex matrix M (N×N) and complex vector b using Gaussian elimination
    N = len(M)
    # Make copies as complex
    A = [[complex(M[i][j]) for j in range(N)] for i in range(N)]
    x = [0j] * N
    rhs = [complex(b[i]) for i in range(N)]
    # Forward elimination with partial pivoting
    for k in range(N):
        # pivot
        piv = max(range(k, N), key=lambda i: abs(A[i][k]))
        if abs(A[piv][k]) == 0:
            raise ValueError("Singular matrix in complex solver")
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            rhs[k], rhs[piv] = rhs[piv], rhs[k]
        # normalize row k
        akk = A[k][k]
        A[k][k] = 1+0j
        for j in range(k+1, N):
            A[k][j] /= akk
        rhs[k] /= akk
        # eliminate below
        for i in range(k+1, N):
            f = A[i][k]
            if f != 0:
                A[i][k] = 0+0j
                for j in range(k+1, N):
                    A[i][j] -= f * A[k][j]
                rhs[i] -= f * rhs[k]
    # Back substitution
    for i in reversed(range(N)):
        s = rhs[i]
        for j in range(i+1, N):
            s -= A[i][j] * x[j]
        x[i] = s  # A[i][i] is 1
    return x

def _get_resname(chain, resnumber, obj="all", state=1):
    resi_str = str(resnumber).strip()
    parts = [f"({obj})", f"resi {resi_str}"]
    ch = ("" if chain is None else str(chain).strip())
    if ch not in ("", "*", "?"):
        parts.append(f"chain {ch}")
    sel = " and ".join(parts)

    model = cmd.get_model(sel, state)
    if not model.atom:
        return None

    resn = (model.atom[0].resn or "").strip().upper()
    return resn or None

def _atoms_for_moiety(obj, chain, resi, resn,geom_mode="default"):
    """Return a list of heavy atoms for the redox-active moiety of a residue."""
    resnU = (resn or "").upper()
    # FAD/FMNs: isoalloxazine
    if resnU in namesFAD:
        if geom_mode=="default":
           names = donacc_names_dict["FADdefault"]
        else:
           names = donacc_names_dict["FADselec"]
    # Aromatics
    elif resnU in donacc_names_dict:
        names = donacc_names_dict[resnU]
    else:
        names = None

    m = cmd.get_model(f"({obj}) and chain {chain} and resi {resi} and resn {resnU}")
    if names:
        sel = [a for a in m.atom if (a.name or "").strip().upper() in names and (a.symbol or "").upper() != "H"]
        if sel: return sel
    # Fallback: heavy-atom sidechain (exclude backbone; drop CB for aromatics)
    out = []
    for a in m.atom:
        nm = (a.name or "").strip().upper()
        if (a.symbol or "").upper() == "H": 
            continue
        if nm in donacc_names_dict["BACKBONE"]:
            continue
        if resnU in {"TRP","TYR","PHE","HIS"} and nm == "CB":
            continue
        out.append(a)
    if out: return out
    # Last resort: all heavy atoms of residue
    return [a for a in m.atom if (a.symbol or "").upper() != "H"]

def _resn_of(obj, chain, resi):
    m = cmd.get_model(f"({obj}) and chain {chain} and resi {resi}")
    return (m.atom[0].resn or "").strip().upper() if m.atom else None

def _min_edge_distance_pymol(obj, node1, node2,geom_mode="default"):
    """Return (Rmin, atomA, atomB) using heavy-atom sets for node1 and node2."""
    ch1, rs1 = node1
    ch2, rs2 = node2
    resn1 = _resn_of(obj, ch1, rs1)
    resn2 = _resn_of(obj, ch2, rs2)
    if not resn1 or not resn2:
        return None, None, None
    
    A = _atoms_for_moiety(obj, ch1, rs1, resn1, geom_mode=geom_mode)
    B = _atoms_for_moiety(obj, ch2, rs2, resn2, geom_mode=geom_mode)
    if not A or not B:
        return None, None, None
    
    coords_A = _coords_of_atoms_pymol(A)
    coords_B = _coords_of_atoms_pymol(B)
    
    # Pairwise distances
    diff = coords_A[:, None, :] - coords_B[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    i_min, j_min = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    dmin = np.sqrt(dist_sq[i_min, j_min])
    
    return dmin, A[i_min], B[j_min]

def _coords_of_atoms_pymol(atoms):
    return np.array([a.coord for a in atoms], dtype=float)

# Utility for orientation factor from raw coords
def _edge_orient_factor_from_coords(donor_points, acceptor_points,
                       orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    Calculate κ for Marcus ET: Normals from full planes; r̂ from minimal atom pair vector.
    
    Args:
        donor_points (list of lists): Donor atoms [[x1,y1,z1], ...] (≥3, coplanar).
        acceptor_points (list of lists): Acceptor atoms [[x1,y1,z1], ...] (≥3, coplanar).
    
    Returns:
        float: κ² (0 ≤ κ² ≤ 1).
    """
    if str(orient_mode).lower() in ("off","","0"):
        return 1.0

    if len(donor_points) < 3 or len(acceptor_points) < 3:
        raise ValueError("At least 3 points required per set.")
    
    donor_arr = np.array(donor_points)
    acceptor_arr = np.array(acceptor_points)
    
    # Full-plane normals (global alignment)
    n_D = _plane_normal_np_coords(donor_points)
    n_A = _plane_normal_np_coords(acceptor_points)
    
    # Find minimal pair: All-to-all distances, get indices of closest
    diffs = acceptor_arr[:, np.newaxis, :] - donor_arr[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)  # (i_min_A, j_min_D)
    
    # Vector from closest donor to acceptor atom
    closest_D = donor_arr[min_idx[1]]  # j_min_D
    closest_A = acceptor_arr[min_idx[0]]  # i_min_A
    r = closest_A - closest_D
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return 0.0  # Degenerate
    r_hat = r / r_norm
    
    # κ = |n_D · r̂| |n_A · (-r̂)|
    kappa = abs(np.dot(n_D, r_hat)) * abs(np.dot(n_A, -r_hat))

    kappa = kappa ** float(orient_pow)
    kappa = max(kappa,float(orient_floor))
    return min(kappa, 1.0)

def _plane_normal_np_coords(points):
    if len(points) < 3:
        raise ValueError("At least 3 points required to define a plane.")
    
    points = np.array(points, dtype=float)
    
    # Center the points (mean)
    center = np.mean(points, axis=0)
    centered = points - center
    
    # SVD decomposition: normal is the singular vector with smallest singular value
    U, S, Vt = np.linalg.svd(centered)
    
    # Last row of Vt (smallest variance direction)
    normal = Vt[-1, :]
    
    # Normalize to unit length
    normal /= np.linalg.norm(normal)
    
    return normal

def _edge_orient_factor_pymol(obj, nodeA, nodeB, geom_mode="default", 
                        orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    Compute orientation factor for edge between nodeA and nodeB in PyMOL object `obj`.

    Wrapper that obtains coordinates for redox moieties of the two nodes,
    then calls `_edge_orient_factor_from_coords()`.

    nodeA, nodeB: tuple (chain, resi)
    """

    chA, rsA = nodeA
    chB, rsB = nodeB

    resnA = _resn_of(obj, chA, rsA)
    resnB = _resn_of(obj, chB, rsB)
    if not resnA or not resnB:
        return 1.0

    A_atoms = _atoms_for_moiety(obj, chA, rsA, resnA, geom_mode=geom_mode)
    B_atoms = _atoms_for_moiety(obj, chB, rsB, resnB, geom_mode=geom_mode)
    if not A_atoms or not B_atoms:
        return 1.0

    coordsA = _coords_of_atoms_pymol(A_atoms)
    coordsB = _coords_of_atoms_pymol(B_atoms)
    return _edge_orient_factor_from_coords(coordsA, coordsB,
                                           orient_mode=orient_mode,
                                           orient_floor=orient_floor,
                                           orient_pow=orient_pow)

def _plane_normal_pymol(atoms):
    """
    Wrapper for _plane_normal_np_coords() that works with PyMOL atom objects.
    Filters coordinates and passes them to the universal implementation.
    """
    coords = _coords_of_atoms_pymol(atoms)
    return _plane_normal_np_coords(coords)

def et_aromatic_metrics(obj, dchain, drsi, achain, arsi, geom_mode="default"):
    """
    Compute metrics for two aromatic side chains: centroid-centroid distance,
    edge-edge min distance, and angle between ring plane normals.
    
    Usage:
      et_aromatic_metrics obj, don_chain, don_resi, acc_chain, acc_resi [, geom_mode]
    
    Visualizes: Residues (sticks), centroids (spheres), closest atoms (labels),
                plane normals (arrows), centroid distance (label).
    
    Supports: TRP, TYR, PHE, HIS.
    """
    import numpy as np
    from math import acos, degrees, pi
    
    # Get resn for donor and acceptor
    resn_d = _resn_of(obj, dchain, drsi)
    resn_a = _resn_of(obj, achain, arsi)
    if resn_d not in namesArom or resn_a not in namesArom:
        print("ERROR: Only aromatic residues supported:",namesArom)
        return
    
    # Get ring atoms for donor
    atoms_d = _atoms_for_moiety(obj, dchain, drsi, resn_d, geom_mode=geom_mode)
    coords_d = _coords_of_atoms_pymol(atoms_d)
    
    # Get ring atoms for acceptor
    atoms_a = _atoms_for_moiety(obj, achain, arsi, resn_a, geom_mode=geom_mode)
    coords_a = _coords_of_atoms_pymol(atoms_a)
    
    if len(coords_d) < 3 or len(coords_a) < 3:
        print("ERROR: Insufficient ring atoms for plane definition (>=3 per ring).")
        return
    
    # Centroids (average of ring atoms)
    cent_d = np.mean(coords_d, axis=0)
    cent_a = np.mean(coords_a, axis=0)
    dist_cent = np.linalg.norm(cent_a - cent_d)
    
    # Edge-edge min distance (min pairwise ring atom distance)
    diff = coords_d[:, None, :] - coords_a[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    min_dist = np.sqrt(np.min(dist_sq))
    i_min, j_min = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    closest_d_atom = atoms_d[i_min]
    closest_a_atom = atoms_a[j_min]
    closest_d_coord = coords_d[i_min]
    closest_a_coord = coords_a[j_min]
    
    # Plane normals (using existing function)
    normal_d = _plane_normal_np_coords(coords_d)
    normal_a = _plane_normal_np_coords(coords_a)
    
    # Angle between normals (acute angle: min(theta, 180-theta))
    dot_prod = np.dot(normal_d, normal_a)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)  # Numerical stability
    angle_rad = acos(abs(dot_prod))  # Absolute for acute angle
    angle_deg = degrees(angle_rad)

    # Orientation factor κ (using existing helper, defaults to 'on')
    kappa = _edge_orient_factor_from_coords(coords_d, coords_a, orient_mode='on')
    
    # Print results
    print("===== Aromatic Ring Metrics =====")
    print(f"Donor: {resn_d} {dchain}:{drsi} | Acceptor: {resn_a} {achain}:{arsi}")
    print(f"Centroid-centroid distance: {dist_cent:.3f} Å")
    print(f"Edge-edge min distance: {min_dist:.3f} Å")
    print(f"  (between {closest_d_atom.name}:{closest_d_atom.resi} and {closest_a_atom.name}:{closest_a_atom.resi})")
    print(f"Ring plane normal (Donor): [{normal_d[0]:.3f}, {normal_d[1]:.3f}, {normal_d[2]:.3f}]")
    print(f"Ring plane normal (Acceptor): [{normal_a[0]:.3f}, {normal_a[1]:.3f}, {normal_a[2]:.3f}]")
    print(f"Angle between plane normals: {angle_deg:.2f}°")
    print("=================================")
    print(f"Orientation factor κ: {kappa:.3f}")
    print("=================================")
    
    # Visualization in PyMOL
    cmd.delete("arom_*")  # Clear previous
    
    # Show residues as sticks, color donor red, acceptor blue
    sel_d = f"({obj}) and chain {dchain} and resi {drsi} and not elem H"
    sel_a = f"({obj}) and chain {achain} and resi {arsi} and not elem H"
    cmd.show("sticks", sel_d)
    cmd.show("sticks", sel_a)
    cmd.color("red", sel_d)
    cmd.color("blue", sel_a)
    # Centroids as pseudoatoms (spheres)
    cmd.pseudoatom("arom_cent_d", pos=cent_d.tolist() )
    cmd.pseudoatom("arom_cent_a", pos=cent_a.tolist() )
    cmd.show("spheres", "arom_cent_d")
    cmd.show("spheres", "arom_cent_a")
    cmd.set("sphere_scale", 0.4, "arom_cent_d or arom_cent_a")
    cmd.color("red", "arom_cent_d")
    cmd.color("blue", "arom_cent_a")
    
    # Closest atoms as pseudoatoms (labels)
    cmd.pseudoatom("arom_closest_d", pos=closest_d_coord.tolist())
    cmd.pseudoatom("arom_closest_a", pos=closest_a_coord.tolist())
    cmd.show("spheres", "arom_closest_d or arom_closest_a")
    cmd.set("sphere_scale", 0.2, "arom_closest_d or arom_closest_a")
    cmd.color("yellow", "arom_closest_d or arom_closest_a")
    #cmd.label("arom_closest_d", "name", closest_d_atom.name)
    #cmd.label("arom_closest_a", "name", closest_a_atom.name)
    
    # Plane normals as arrows (scale=2 Å)
    scale = 2.0
    #cmd.load_cgo(_arrow("arom_normal_d", cent_d.tolist(), (cent_d + normal_d * scale).tolist(), rgb=(1,0,0)), "arom_normal_d")
    #cmd.load_cgo(_arrow("arom_normal_a", cent_a.tolist(), (cent_a + normal_a * scale).tolist(), rgb=(0,0,1)), "arom_normal_a")
    
    # Centroid-centroid distance label
    cmd.distance("arom_dist_cent", "arom_cent_d", "arom_cent_a")
    cmd.color("white", "arom_dist_cent")
    cmd.set("dash_width", 3, "arom_dist_cent")
    
    # Edge-edge distance label
    cmd.distance("arom_dist_edge", "arom_closest_d", "arom_closest_a")
    cmd.color("yellow", "arom_dist_edge")
    
    print("Visualization loaded: Check PyMOL viewer for rings, centroids, normals, and distances.")

# Register the command
cmd.extend("et_aromatic_metrics", et_aromatic_metrics)

# ---------------------- visuals pymol helpers ---------------------------
def _safe_delete(name):
    if name in cmd.get_names('all'):
        cmd.delete(name)

def _arrow(name, start, end, rgb=(1.0,0.6,0.0), radius=0.24, head_len=0.8, head_radius=0.45):
    sx,sy,sz = start; ex,ey,ez = end
    vx,vy,vz = (ex-sx, ey-sy, ez-sz)
    L = max((vx*vx+vy*vy+vz*vz)**0.5, 1e-6)
    ux,uy,uz = (vx/L, vy/L, vz/L)
    hx,hy,hz = (ex-ux*head_len, ey-uy*head_len, ez-uz*head_len)
    r,g,b = rgb
    obj = [
        cgo.CYLINDER, sx,sy,sz, hx,hy,hz, radius, r,g,b, r,g,b,
        cgo.CONE, hx,hy,hz, ex,ey,ez, head_radius, 0.0, r,g,b, r,g,b, 1.0, 0.0
    ]
    _safe_delete(name)
    cmd.load_cgo(obj, name)

def _show_residue(obj, chain, resi, color):
    sel = f"({obj}) and chain {chain} and resi {resi} and not elem H"
    cmd.show("sticks", sel)
    cmd.color(color, sel)

# ---------------------- parsing helpers --------------------------
def _extract_mode_and_args(args):
    """If first token is 'default' or 'selec_atoms', return (geom_mode, rest_of_args), else ('default', args)."""
    if len(args) > 0 and str(args[0]).lower() in ("default", "selec_atoms"):
        return str(args[0]).lower(), list(args[1:])
    return "default", list(args)

def _get_arg(args, idx, default):
    if idx >= len(args): return default
    s = str(args[idx]).strip()
    return s if s != "" else default

def _parse_bridge_list(bridges_str):
    # bridges_str like "A:359+A:382" or "A:359" or ""
    s = str(bridges_str).strip()
    if not s: return []
    out = []
    for tok in s.split("+"):
        tok = tok.strip()
        if not tok: continue
        if ":" not in tok:
            raise ValueError(f"Bridge token '{tok}' must be 'Chain:Resi'")
        ch, rs = tok.split(":", 1)
        out.append((ch.strip(), rs.strip()))
    return out

def _parse_float_list(csv_str, N_expected=None):
    s = str(csv_str).strip()
    if not s:
        return []
    vals = [float(x.strip()) for x in s.replace(";",",").split(",") if x.strip()!=""]
    if N_expected is not None and len(vals) != N_expected:
        raise ValueError(f"Expected {N_expected} values, got {len(vals)}")
    return vals

def _is_float_token(s):
    try:
        float(str(s).strip())
        return True
    except Exception:
        return False

# ---------------------- Marcus and couplings ----------------------
def _compute_et_rate_unified(engine,
                             donor_source,
                             acceptor_source,
                             geom_mode="default",
                             orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    Universal geometry → rate workflow for single-step ET.
    Works for PyMOL object mode or raw coordinate arrays mode.
    """
    a_best, b_best = None, None  # default

    if isinstance(donor_source, np.ndarray) and isinstance(acceptor_source, np.ndarray):
        # Coordinates mode
        coords_donor = donor_source
        coords_acceptor = acceptor_source
        diff = coords_donor[:, None, :] - coords_acceptor[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        if dist_sq.size == 0:
            raise ValueError("No distances computed (empty coordinate arrays)")
        i_min, j_min = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        R_eff = np.sqrt(dist_sq[i_min, j_min])
        f_orient = _edge_orient_factor_from_coords(coords_donor, coords_acceptor,
                                                   orient_mode=orient_mode,
                                                   orient_floor=orient_floor,
                                                   orient_pow=orient_pow)

    elif isinstance(donor_source, tuple) and isinstance(acceptor_source, tuple):
        # PyMOL mode
        obj_name, dch, drs = donor_source
        _, ach, ars = acceptor_source
        R_eff, a_best, b_best = _min_edge_distance_pymol(obj_name, (dch, drs), (ach, ars),
                                                        geom_mode=geom_mode)
        if R_eff is None:
            raise ValueError("Distance not found for donor/acceptor pair")

        f_orient = _edge_orient_factor_pymol(obj_name, (dch, drs), (ach, ars),
                                             geom_mode=geom_mode,
                                             orient_mode=orient_mode,
                                             orient_floor=orient_floor,
                                             orient_pow=orient_pow)

        coords_donor = _coords_of_atoms_pymol(_atoms_for_moiety(
            obj_name, dch, drs, _resn_of(obj_name, dch, drs),
            geom_mode))
        coords_acceptor = _coords_of_atoms_pymol(_atoms_for_moiety(
            obj_name, ach, ars, _resn_of(obj_name, ach, ars),
            geom_mode))
    else:
        raise TypeError("donor_source/acceptor_source must be both np.ndarrays or both tuples")

    geom_dict = {'R_eff': R_eff}
    Hab = 0
    # Only compute Hab for theories that need it
    if not isinstance(engine, EmpiricalMarkusEngine):
        H0   = engine.params.get('H0', DEF_H0)
        beta = engine.params.get('beta', DEF_BETA)
        R0   = engine.params.get('R0', DEF_R0)
        Hab_geom = _Vpair(R_eff, H0, beta, R0)
        Hab = Hab_geom * f_orient
        geom_dict['Hab_eV'] = Hab

    k = engine.compute_rate(Hab, geom_dict)
    tau_ps = 1e12 / k if k > 0 else float('inf')
    return k, tau_ps, geom_dict, f_orient, Hab, coords_donor, coords_acceptor, a_best, b_best

def _Vpair(R, H0, beta, R0):
    """Distance-based electronic coupling (exponential decay model)."""
    return H0 * np.exp(-beta * (R - R0))

def _marcus_rate(Hab_eV, lambda_eV, deltaG_eV, T_K):
    """Standard Marcus Theory calculation"""
    pref = (2.0*np.pi) / hbar_eV_s
    denom = np.sqrt(4.0*np.pi*lambda_eV*kB_eV_per_K*T_K)
    expo  = -((deltaG_eV + lambda_eV)**2) / (4.0*lambda_eV*kB_eV_per_K*T_K)
    return pref * (Hab_eV*Hab_eV) * (1.0/denom) * np.exp(expo)

def _Hab_needed_for_k(k_target, lambda_eV, dG_eV, T_K):
  denom = np.sqrt(4.0*np.pi*lambda_eV*kB_eV_per_K*T_K)
  pref = (2.0*np.pi)/hbar_eV_s
  expo = np.exp(-((dG_eV + lambda_eV)**2)/(4.0*lambda_eV*kB_eV_per_K*T_K))
  Hab = np.sqrt(k_target * denom / (pref * expo))
  return Hab

def _Hab_from_distance(R_A, H0, beta, R0):
  return H0 * np.exp(-beta*(R_A - R0))

def _mlj_rate(Hab_eV, lambda_s_eV, deltaG_eV, T_K, S=1.0, hw_eV=0.18, vmax=10):
    """
    Marcus–Levich–Jortner (MLJ) electron transfer rate.
    
    Parameters:
      Hab_eV : float
          Electronic coupling in eV
      lambda_s_eV : float
          Classical (solvent) reorganization energy [eV]
      deltaG_eV : float
          Reaction free energy [eV]
      T_K : float
          Temperature [K]
      S : float
          Huang–Rhys factor (dimensionless, ~0.5–2 typical)
      hw_eV : float
          Vibrational quantum energy (hbar*omega) [eV], e.g. 0.18 eV ≈ 1450 cm^-1
      vmax : int
          Maximum vibrational quantum number to include in sum

    Returns:
      k [s^-1]
    """
    pref = (2.0*np.pi) / hbar_eV_s
    ksum = 0.0
    for v in range(vmax+1):
        # Franck–Condon factor (Poisson distribution)
        FC = np.exp(-S) * (S**v) / factorial(v)
        # Gaussian nuclear term (solvent reorganization)
        denom = np.sqrt(4.0*np.pi*lambda_s_eV*kB_eV_per_K*T_K)
        expo  = -((deltaG_eV + lambda_s_eV + v*hw_eV)**2) / (4.0*lambda_s_eV*kB_eV_per_K*T_K)
        term  = FC * (1.0/denom) * np.exp(expo)
        ksum += term
    return pref * (Hab_eV*Hab_eV) * ksum

def _mlj_rate_multi(Hab_eV, lambda_s_eV, deltaG_eV, T_K,
                    S_list, hw_list, vmax_list):
    """
    Multi-mode Marcus–Levich–Jortner rate.

    Parameters
    ----------
    Hab_eV : float
        Electronic coupling (eV)
    lambda_s_eV : float
        Classical (solvent) reorganization energy (eV)
    deltaG_eV : float
        Reaction free energy (eV)
    T_K : float
        Temperature (K)
    S_list : list of floats
        Huang–Rhys factors for each mode.
    hw_list : list of floats
        Vibrational quantum energies ℏω (eV) for each mode.
    vmax_list : list of ints
        Maximum v quanta to include for each mode.

    Returns
    -------
    k : float
        ET rate (s^-1)
    """
    pref = (2.0 * np.pi) / hbar_eV_s
    ksum = 0.0

    # Generate all v_i combinations using itertools.product
    ranges = [range(vmax_i + 1) for vmax_i in vmax_list]
    for v_tuple in itertools.product(*ranges):
        # Franck–Condon product term
        FC_prod = 1.0
        vib_sum_energy = 0.0
        for idx, v_i in enumerate(v_tuple):
            S_i = S_list[idx]
            hw_i = hw_list[idx]
            FC_prod *= np.exp(-S_i) * (S_i**v_i) / factorial(v_i)
            vib_sum_energy += v_i * hw_i

        # Marcus nuclear Gaussian
        denom = np.sqrt(4.0 * np.pi * lambda_s_eV * kB_eV_per_K * T_K)
        expo = -((deltaG_eV + lambda_s_eV + vib_sum_energy)**2) / \
               (4.0 * lambda_s_eV * kB_eV_per_K * T_K)
        term = FC_prod * (1.0 / denom) * np.exp(expo)
        ksum += term

    return pref * (Hab_eV * Hab_eV) * ksum

def _effective_rate_hopping_chain(V_edges, dG_steps, lambda_eV, T_K):
    """
    Compute effective hopping rate from donor (state 0) to acceptor (absorbing, last state)
    via MFPT on a linear chain with nearest-neighbor steps only.
    """
    N_edges = len(V_edges)
    if len(dG_steps) != N_edges:
        raise ValueError(f"dG_steps length {len(dG_steps)} must equal number of edges {N_edges}")
    N_nodes = N_edges + 1

    # Build generator matrix K
    K = np.zeros((N_nodes, N_nodes), dtype=float)
    for i in range(N_edges):
        V = float(V_edges[i])
        dG = float(dG_steps[i])
        kf = _marcus_rate(V, lambda_eV, dG, T_K)
        kb = _marcus_rate(V, lambda_eV, -dG, T_K)
        K[i, i+1] = kf
        if i+1 < N_nodes-1:   # no back from absorbing state
            K[i+1, i] = kb

    # Diagonal entries for transient states
    for i in range(N_nodes-1):
        K[i, i] = -np.sum(K[i, :]) + K[i, i]

    # Extract Q (top-left block, transient submatrix)
    N_tr = N_nodes - 1
    Q = K[:N_tr, :N_tr]
    rhs = np.ones((N_tr,), dtype=float)

    # Solve MFPT system: (-Q) τ = 1
    tau = np.linalg.solve(-Q, rhs)
    tau0 = float(tau[0])
    return 1.0 / tau0 if tau0 > 0.0 else 0.0


# ---------------------- redox-to-energy mapping -------------------
# We map redox potentials (V vs. NHE) to site energies in eV by ε = -E (electron convention).
# Tunneling energy Etun ≈ (εD + εA)/2 = -(Eox_D + Ered_A)/2.
# Then ΔBi = εBi - Etun = -Ered_Bi + 0.5*(Eox_D + Ered_A). All in eV.

def _site_gap_from_potentials(Eox_D, Ered_A, Ered_Bi):
    return -float(Ered_Bi) + 0.5*(float(Eox_D) + float(Ered_A))

# ---------------------- helpers for Green's function evaluation ----------------------------

def _run_chain(obj, donor, bridges, acceptor, lambda_eV, dG, T, H0, beta, R0, deltaB_list, geom_mode="default", orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    Compute Hab_eff for D -- B1 -- ... -- Bn -- A and Marcus rate.
    Visualize residues, arrows, and distances. Print a detailed report.
    """
    cmd.remove("hydro")
    # Gather node list
    nodes = [donor] + list(bridges) + [acceptor]

    # Show residues
    colors = ["red"] + ["yellow"]*len(bridges) + ["blue"]
    for (ch,rs), col in zip(nodes, colors):
        _show_residue(obj, ch, rs, col)

    # For each segment, get edge-edge distance and closest atoms
    segs = []
    for i in range(len(nodes)-1):
        R, a1, a2 = _min_edge_distance_pymol(obj, nodes[i], nodes[i+1], geom_mode)
        if R is None:
            print(f"ERROR: Could not determine distance for segment {i} ({nodes[i]} -> {nodes[i+1]}).")
            return
        segs.append((R,a1,a2))

    # Draw arrows and distance labels
    # Clear prior chain visuals
    for i in range(len(nodes)-1):
        _safe_delete(f"BR_arrow_{i}")
        _safe_delete(f"BR_dist_{i}")
        _safe_delete(f"BR_pA_{i}")
        _safe_delete(f"BR_pB_{i}")

    for i,(R,a1,a2) in enumerate(segs):
        _arrow(f"BR_arrow_{i}", a1.coord, a2.coord, rgb=(1.0,0.6,0.0))
        # Pseudoatoms at the closest heavy atoms
        cmd.pseudoatom(f"BR_pA_{i}", pos=a1.coord)
        cmd.pseudoatom(f"BR_pB_{i}", pos=a2.coord)
        cmd.set("sphere_scale", 0.3, f"BR_pA_{i}")
        cmd.set("sphere_scale", 0.3, f"BR_pB_{i}")
        cmd.distance(f"BR_dist_{i}", f"BR_pA_{i}", f"BR_pB_{i}")

    # Compute edge couplings and Hab_eff (McConnell product form)
    Vedges = []
    for i in range(len(nodes)-1):
        R, _, _ = segs[i]
        f = _edge_orient_factor_pymol(obj, nodes[i], nodes[i+1], geom_mode=geom_mode, orient_mode=orient_mode, orient_floor=orient_floor, orient_pow=orient_pow)
        Vedges.append(_Vpair(R, H0, beta, R0) * f)
    # ΔB list length must equal number of bridges (n_bridges = len(nodes)-2)
    if len(deltaB_list) != max(0, len(nodes)-2):
        print(f"ERROR: deltaB_list length {len(deltaB_list)} must equal number of bridges {max(0,len(nodes)-2)}")
        return

    # product over all edges
    Vprod = 1.0
    for V in Vedges:
        Vprod *= V
    # product over all bridge gaps
    Dprod = 1.0
    for dB in deltaB_list:
        Dprod *= float(dB)
    Hab_eff = Vprod / (Dprod if Dprod != 0.0 else 1e-30)

    k = _marcus_rate(Hab_eff, lambda_eV, dG, T)
    tau_ps = 1e12/k if k>0 else float('inf')

    # Report
    print("===== Bridge-mediated ET (super-exchange, McConnell) =====")
    print(f"Object: {obj}")
    lab_nodes = []
    for ch,rs in nodes:
        resn = _resn_of(obj, ch, rs) or "UNK"
        lab_nodes.append(f"{resn} {ch}/{rs}")
    print("Path: " + "  ->  ".join(lab_nodes))
    print(f"Params: λ={lambda_eV:.2f} eV, ΔG={dG:+.2f} eV, T={T:.2f} K, H0={H0:.3f} eV, β={beta:.2f} Å^-1, R0={R0:.2f} Å")
    if len(deltaB_list):
        print("Bridge gaps ΔB_i (eV): " + ", ".join(f"{x:.2f}" for x in deltaB_list))
    else:
        print("No bridge gaps (direct coupling)")

    print("Segments:")
    for i,(R,_,_) in enumerate(segs):
        f = Vedges[i] / max(_Vpair(R, H0, beta, R0), 1e-10)  # Avoid div0
        print(f"  {lab_nodes[i]} -> {lab_nodes[i+1]}: R={R:.2f} Å, κ²={f:.3f}, V={Vedges[i]*1000:.2f} meV")

    print(f"Hab_eff = {Hab_eff*1000:.2f} meV")
    print(f"k = {k:.3e} s^-1   τ = {tau_ps:.2f} ps")
    print("==========================================================")
    return

def et_bridge_chain_gf_core(distance_func, obj, dCh, dRs, aCh, aRs,
                            bridges, Emode, emode_args,
                            lambda_eV, dG, T, H0, beta, R0,
                            gamma, connect,
                            geom_mode, orient_mode, orient_floor, orient_pow):
    N = len(bridges)
    eps = []

    # Energy mode setup
    if Emode == "pot":
        EoxD, EredA, Ereds = emode_args
        Etun = -0.5 * (EoxD + EredA)
        eps  = [-float(Eb) for Eb in Ereds]
    elif Emode == "const":
        Etun = emode_args[0]
        deltaBs = emode_args[1:]
        if len(deltaBs) == 1:
            deltaBs = deltaBs * N
        eps = [Etun + dB for dB in deltaBs]
    else:
        raise ValueError(f"Unknown Emode '{Emode}'")

    # Build couplings
    VDB = [distance_func(obj, (dCh,dRs), bi, geom_mode, orient_mode, orient_floor, orient_pow, H0, beta, R0)[1]
           for bi in bridges]
    VBA = [distance_func(obj, bi, (aCh,aRs), geom_mode, orient_mode, orient_floor, orient_pow, H0, beta, R0)[1]
           for bi in bridges]
    Tij = [[0.0 for _ in range(N)] for __ in range(N)]
    if connect == "allpairs":
        for iB in range(N):
            for jB in range(iB+1, N):
                _, t = distance_func(obj, bridges[iB], bridges[jB], geom_mode, orient_mode, orient_floor, orient_pow, H0, beta, R0)
                Tij[iB][jB] = t
                Tij[jB][iB] = t
    else:
        for iB in range(N-1):
            _, t = distance_func(obj, bridges[iB], bridges[iB+1], geom_mode, orient_mode, orient_floor, orient_pow, H0, beta, R0)
            Tij[iB][iB+1] = t
            Tij[iB+1][iB] = t

    # Build M-matrix
    M = [[0j for _ in range(N)] for __ in range(N)]
    Ecomplex = complex(Etun, gamma)
    for iB in range(N):
        for jB in range(N):
            if iB == jB:
                M[iB][jB] = Ecomplex - complex(eps[iB], 0.0)
            else:
                M[iB][jB] = -complex(Tij[iB][jB], 0.0)

    x = _complex_solve(M, [complex(v, 0.0) for v in VBA])
    Hab_eff = sum(complex(VDB[i],0.0) * x[i] for i in range(N))
    return _marcus_rate(abs(Hab_eff), lambda_eV, dG, T)

def et_bridge_chain_hop_core(distance_func, obj, dCh, dRs, aCh, aRs,
                             bridges, Emode, emode_args,
                             lambda_eV, T, H0, beta, R0,
                             geom_mode, orient_mode, orient_floor, orient_pow):
    N_edges = len(bridges) + 1

    if Emode == "pot":
        EoxD, EredA, Ereds = emode_args
        eps_D = -EoxD
        eps_A = -EredA
        eps_B = [-Eb for Eb in Ereds]
        eps = [eps_D] + eps_B + [eps_A]
        dG_steps = [eps[k+1] - eps[k] for k in range(len(eps)-1)]
    elif Emode == "dg":
        vals = emode_args
        if len(vals) == 1:
            dG_steps = [vals[0]] * N_edges
        else:
            dG_steps = vals[:N_edges]
    else:
        raise ValueError(f"Unknown Emode '{Emode}' in hop")

    # Couplings per edge
    nodes = [(dCh,dRs)] + bridges + [(aCh,aRs)]
    V_edges = [distance_func(obj, nodes[k], nodes[k+1], geom_mode, orient_mode, orient_floor, orient_pow, H0, beta, R0)[1]
               for k in range(len(nodes)-1)]

    return _effective_rate_hopping_chain(V_edges, dG_steps, lambda_eV, T)

# ---------------------- PyMOL command extensions --------------------------
def estimate_lambda_outer(R_DA_A, aD_A=4.0, aA_A=4.0, eps_s=4.0, eps_op=2.0):
    # Avoid division by zero if R_DA_A is very small
    R = max(float(R_DA_A), 1.0e-6)

    # Geometric term
    expr = (1.0/(2.0*aD_A)) + (1.0/(2.0*aA_A)) - (1.0/R)

    # Dielectric term
    diel_term = (1.0/eps_op) - (1.0/eps_s)

    return _COULOMB_CONST * expr * diel_term


def calibrate_Marcus_to_k(obj, don_chain, don_resi, acc_chain, acc_resi,
                          theory='marcus',  # e.g., 'marcus', 'mlj', 'redfield', 'tsh'
                          engine_params=None,  # Dict of params like {'lambda_eV': 0.7, ...}
                          k_target=1.25e12,
                          fit_params=('H0',),  # List/tuple: 'H0', 'lambda', 'beta', or combos
                          plot=None,  # None, '1D', '2D'
                          sweep_range_H0=(0.01, 0.2),
                          sweep_range_lambda=(0.3, 1.2),
                          sweep_range_beta=(0.1, 2.0),
                          fit_range_H0=(0.001, 0.5),
                          fit_range_lambda=(0.1, 2.0),
                          fit_range_beta=(0.1, 2.0),
                          geom_mode="default",
                          orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                          visualize="off"):
    """
    Calibrate/fit ET parameters to target rate using any ETEngine.
    
    - Inversion branch: For single param + plot=None, direct solve (fast).
    - Fitting branch: For multi-param or plot=1D/2D, scipy.optimize + sweeps.
    
    engine_params: Dict for engine init (e.g., {'lambda_eV': 0.70, 'deltaG_eV': -0.50, 'T_K': 298.15}).
    Defaults to Marcus if theory='marcus'.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    if engine_params is None:
        engine_params = {
            'lambda_eV': DEF_LAMBDA, 'deltaG_eV': DEF_DELTAG, 'T_K': DEF_TK,
            'H0': DEF_H0, 'beta': DEF_BETA, 'R0': DEF_R0
        }
    
    # Instantiate engine based on theory
    try:
        engine_class = _engine_map[theory.lower()]
    except KeyError:
        raise ValueError(f"Unknown theory '{theory}'. Available: {list(_engine_map.keys())}")
    # MLJ special-case to ensure required lists
    if theory.lower() == 'mlj':
        engine_params.setdefault('S_list', [1.0])
        engine_params.setdefault('hw_list', [0.18])
        engine_params.setdefault('vmax_list', [10])
    engine = engine_class(engine_params) 
   
    # --- Initial geometric setup ---
    k_init, tau_init, geom_dict, f_orient, Hab, donor_coords, acceptor_coords, a_best, b_best = _compute_et_rate_unified(
        engine,(obj, don_chain, don_resi), (obj, acc_chain, acc_resi),
        geom_mode=geom_mode, orient_mode=orient_mode,
        orient_floor=orient_floor, orient_pow=orient_pow )
    
    # Visualization
    if visualize == "on"and a_best and b_best:
        # Existing viz code (pseudoatoms, arrow, distance)
        _safe_delete("DA_don"); _safe_delete("DA_acc"); _safe_delete("DA_arrow"); _safe_delete("DA_dist")
        cmd.pseudoatom("DA_don", pos=a_best.coord); cmd.color("blue", "DA_don")
        cmd.show("spheres", "DA_don"); cmd.set("sphere_scale", 0.4, "DA_don")
        cmd.pseudoatom("DA_acc", pos=b_best.coord); cmd.color("orange", "DA_acc")
        cmd.show("spheres", "DA_acc"); cmd.set("sphere_scale", 0.4, "DA_acc")
        _arrow("DA_arrow", a_best.coord, b_best.coord, rgb=(1.0,0.6,0.0))
        cmd.distance("DA_dist", "DA_don", "DA_acc")
    
    fit_params = list(fit_params)  # Ensure list
    
    # === INVERSION BRANCH: Single param, no plot ===
    if len(fit_params) == 1 and plot is None:
        param = fit_params[0]
        H0 = engine_params.get('H0', DEF_H0)
        beta = engine_params.get('beta', DEF_BETA)
        R0 = engine_params.get('R0', DEF_R0)
        lambda_eV = engine_params.get('lambda_eV', DEF_LAMBDA)
        dG = engine_params.get('deltaG_eV', DEF_DELTAG)
        T = engine_params.get('T_K', DEF_TK)
        
        print(f"=== Direct inversion for {param} ({theory.upper()}) ===")
        print(f"R_eff = {geom_dict['R_eff']:.3f} Å, f_orient = {f_orient:.3f}")
        
        if param == 'H0':
            # Solve for H0: Hab_needed = sqrt( k_target * denom / (pref * expo) )
            # Then H0 = Hab_needed / (f_orient * exp(-beta (geom_dict['R_eff'] - R0)))
            denom = np.sqrt(4.0 * np.pi * lambda_eV * kB_eV_per_K * T)
            pref = (2.0 * np.pi) / hbar_eV_s
            expo = np.exp(-((dG + lambda_eV)**2) / (4.0 * lambda_eV * kB_eV_per_K * T))
            Hab_needed = np.sqrt(k_target * denom / (pref * expo))
            H0_needed = Hab_needed / (f_orient * np.exp(-beta * (geom_dict['R_eff'] - R0)))
            print(f"Required Hab = {Hab_needed*1000:.2f} meV")
            print(f"Set H0 ≈ {H0_needed:.4f} eV")
            engine_params['H0'] = H0_needed
            
        elif param == 'lambda':
            # Solve for lambda: Numerical root-find for rate_diff(lambda) = 0
            Hab_geom = H0 * np.exp(-beta * (geom_dict['R_eff'] - R0)) * f_orient
            prefactor = (2 * np.pi / hbar_eV_s) * (Hab_geom**2)
            def rate_diff(lambda_eV):
                if lambda_eV <= 0: return np.inf
                denom = np.sqrt(4 * np.pi * lambda_eV * kB_eV_per_K * T)
                expo = np.exp(-((dG + lambda_eV)**2) / (4 * lambda_eV * kB_eV_per_K * T))
                return prefactor * (1 / denom) * expo - k_target
            from scipy.optimize import brentq
            try:
                lambda_needed = brentq(rate_diff, 1e-5, 10.0)
            except:
                lambda_needed = 0.7  # Fallback
            print(f"Set λ ≈ {lambda_needed:.4f} eV")
            engine_params['lambda_eV'] = lambda_needed
            
        elif param == 'beta':
            # Solve for beta: beta = -log(Hab_needed / (H0 * f_orient)) / (geom_dict['R_eff'] - R0)
            denom = np.sqrt(4.0 * np.pi * lambda_eV * kB_eV_per_K * T)
            pref = (2.0 * np.pi) / hbar_eV_s
            expo = np.exp(-((dG + lambda_eV)**2) / (4.0 * lambda_eV * kB_eV_per_K * T))
            Hab_needed = np.sqrt(k_target * denom / (pref * expo))
            beta_needed = -np.log(Hab_needed / (H0 * f_orient)) / (geom_dict['R_eff'] - R0)
            print(f"Required Hab = {Hab_needed*1000:.2f} meV")
            print(f"Set β ≈ {beta_needed:.4f} Å^-1")
            engine_params['beta'] = beta_needed
        
        # Recompute final k
        Hab = H0 * np.exp(-beta * (geom_dict['R_eff'] - R0)) * f_orient  # Update with new param
        k_fit = engine.compute_rate(Hab, geom_dict)
        print(f"Predicted k = {k_fit:.3e} s^-1 (target {k_target:.3e})")
        return engine_params  # Updated params
    
    # === FITTING BRANCH: Multi-param or plot ===
    def objective(x):
        trial_params = engine_params.copy()
        for i, key in enumerate(fit_params):
            trial_params[key if key != 'lambda' else 'lambda_eV'] = x[i]
        trial_engine = engine_class(trial_params)
        k_trial, _, geom_trial, _, _, _, _, _, _ = _compute_et_rate_unified(
            trial_engine,(obj, don_chain, don_resi),(obj, acc_chain, acc_resi),
            geom_mode=geom_mode, orient_mode=orient_mode,
            orient_floor=orient_floor, orient_pow=orient_pow )
        return abs(np.log10(k_trial) - np.log10(k_target)) if k_trial > 0 else np.inf

    # Initial guess and bounds
    p0 = []
    bounds = []
    range_map = {'H0': sweep_range_H0, 'lambda': sweep_range_lambda, 'beta': sweep_range_beta}
    bound_map = {'H0': fit_range_H0, 'lambda': fit_range_lambda, 'beta': fit_range_beta}
    for p in fit_params:
        p0.append(engine_params.get('H0' if p=='H0' else 'lambda_eV' if p=='lambda' else 'beta', DEF_H0))
        bounds.append(bound_map[p])
    
    res = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')
    if not res.success:
        print(f"Fit failed: {res.message}")
        return
    
    # Unpack fitted
    fitted = {}
    idx = 0
    if 'H0' in fit_params:
        fitted['H0'] = res.x[idx]; idx += 1
    if 'lambda' in fit_params:
        fitted['lambda'] = res.x[idx]; idx += 1
    if 'beta' in fit_params:
        fitted['beta'] = res.x[idx]; idx += 1
    engine_params.update({k: v for k, v in fitted.items() if k != 'lambda'})  # lambda -> lambda_eV
    if 'lambda' in fitted:
        engine_params['lambda_eV'] = fitted['lambda']
    
    # Final Hab and k
    Hab_fit = engine_params['H0'] * np.exp(-engine_params['beta'] * (geom_dict['R_eff'] - engine_params['R0'])) * f_orient
    k_fit = engine.compute_rate(Hab_fit, geom_dict)
    
    print(f"=== Calibration & Fit ({theory.upper()}) ===")
    print(f"R_eff = {geom_dict['R_eff']:.3f} Å")
    print(f"Fit params: {fit_params}")
    for p, v in fitted.items():
        print(f"{p}_fit = {v:.5f} {'eV' if p in ('H0', 'lambda') else 'Å^-1'}")
    print(f"orient_factor = {f_orient:.3f}")
    print(f"Predicted k_fit = {k_fit:.3e} s^-1 (target {k_target:.3e} s^-1)")
    
    # === PLOTTING ===
    if plot == 'Yes' and len(fit_params) == 1:
        param = fit_params[0]
        xs = np.linspace(*range_map[param], 100)
        ys = []
        for x in xs:
            trial_p = engine_params.copy()
            trial_p['H0' if param=='H0' else 'lambda_eV' if param=='lambda' else 'beta'] = x
            trial_engine = engine.__class__(trial_p)
            trial_Hab = trial_p['H0'] * np.exp(-trial_p['beta'] * (geom_dict['R_eff'] - trial_p['R0'])) * f_orient
            ys.append(trial_engine.compute_rate(trial_Hab, geom_dict))
        plt.figure()
        plt.plot(xs, ys)
        plt.axvline(fitted[param], color='green', linestyle=':')
        plt.axhline(k_target, color='red', linestyle='--')
        plt.xlabel(param)
        plt.ylabel("Rate (s$^{-1}$)")
        plt.yscale('log')
        plt.title(f"1D Sweep: {param} ({theory.upper()})")
        plt.show()
    elif plot == 'Yes' and len(fit_params) == 2:
        paramX, paramY = fit_params
        xs = np.linspace(*range_map[paramX], 50)
        ys = np.linspace(*range_map[paramY], 50)
        Z = np.zeros((len(ys), len(xs)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                trial_p = engine_params.copy()
                trial_p[paramX if paramX != 'lambda' else 'lambda_eV'] = x
                trial_p[paramY if paramY != 'lambda' else 'lambda_eV'] = y
                trial_engine = engine.__class__(trial_p)
                trial_Hab = trial_p['H0'] * np.exp(-trial_p['beta'] * (geom_dict['R_eff'] - trial_p['R0'])) * f_orient
                Z[j, i] = trial_engine.compute_rate(trial_Hab, geom_dict)
        plt.figure()
        cs = plt.contourf(xs, ys, np.log10(Z), levels=50, cmap='viridis')
        plt.colorbar(cs, label="log10 Rate [s$^{-1}$]")
        plt.scatter(fitted[paramX], fitted[paramY], color='red', label="Fit")
        plt.xlabel(paramX); plt.ylabel(paramY)
        plt.title(f"2D Contour: {paramX} vs {paramY} ({theory.upper()})")
        plt.legend(); plt.show()
    
    return engine_params  # Updated fitted params

cmd.extend("calibrate_Marcus_to_k", calibrate_Marcus_to_k)


def et_rate_marcus(*args, _self=cmd, theory='marcus', engine_params=None, **kwargs):
    geom_mode, args = _extract_mode_and_args(args)
    if len(args) < 5:
        print(USAGE_RATE); return

    # Detect if engine_params from GUI (dict in args[5])
    gui_params = None
    if len(args) > 5 and isinstance(args[5], dict):
        gui_params = args[5]
        # Truncate args to geometry + orient (from kwargs or args tail)
        args = args[:5] + list(args[6:]) if len(args) > 6 else args[:5]
        orient_mode = _get_arg(args, 5, "off") if len(args) > 5 else kwargs.get('orient_mode', 'off')
        orient_floor = float(_get_arg(args, 6, DEF_Floor)) if len(args) > 6 else float(kwargs.get('orient_floor', DEF_Floor))
        orient_pow = float(_get_arg(args, 7, DEF_POW)) if len(args) > 7 else float(kwargs.get('orient_pow', DEF_POW))
    else:
        # Backward-compatible parsing
        try:
            obj, don_chain, don_resi, acc_chain, acc_resi = [str(a).strip() for a in args[:5]]
            orient_mode  = _get_arg(args, 11, "off")
            orient_floor = float(_get_arg(args, 12, DEF_Floor))
            orient_pow   = float(_get_arg(args, 13, DEF_POW))
            engine_params = {
                'lambda_eV': float(_get_arg(args, 5, DEF_LAMBDA)), 'deltaG_eV': float(_get_arg(args, 6, DEF_DELTAG)), 
                'T_K': float(_get_arg(args, 7, DEF_TK)),           'H0': float(_get_arg(args, 8, DEF_H0)), 
                'beta': float(_get_arg(args, 9, DEF_BETA)),        'R0': float(_get_arg(args,10, DEF_R0))
            }
        except Exception as e:
            print(f"Argument parsing error: {e}")
            print(USAGE_RATE); return

    # Use gui_params if detected, else engine_params (from kwarg, for future)
    engine_params = gui_params if gui_params is not None else engine_params
    if engine_params is None:
        engine_params = {'lambda_eV': DEF_LAMBDA, 'deltaG_eV': DEF_DELTAG, 'T_K': DEF_TK,
                        'H0': DEF_H0, 'beta': DEF_BETA, 'R0': DEF_R0}
    
    # Instantiate engine
    # Instantiate engine based on theory
    try:
        engine_class = _engine_map[theory.lower()]
    except KeyError:
        raise ValueError(f"Unknown theory '{theory}'. Available: {list(_engine_map.keys())}")
    engine = engine_class(engine_params) 

    obj, don_chain, don_resi, acc_chain, acc_resi = [str(a).strip() for a in args[:5]]
    k, tau_ps, geom_dict, f_orient, Hab, donor_coords, acceptor_coords,a_best,b_best = _compute_et_rate_unified(
        engine, (obj, don_chain, don_resi), (obj, acc_chain, acc_resi),
        geom_mode=geom_mode, orient_mode=orient_mode,
        orient_floor=orient_floor, orient_pow=orient_pow)
    tau_ps = 1e12 / k if k > 0 else float('inf')

    # Viz (unchanged)
    _safe_delete("DA_arrow"); _safe_delete("DA_dist")
    _safe_delete("DA_don"); _safe_delete("DA_acc")
    if a_best and b_best:
        cmd.pseudoatom("DA_don", pos=a_best.coord)
        cmd.color("blue", "DA_don")
        cmd.show("spheres", "DA_don")
        cmd.set("sphere_scale", 0.4, "DA_don")

        cmd.pseudoatom("DA_acc", pos=b_best.coord)
        cmd.color("orange", "DA_acc")
        cmd.show("spheres", "DA_acc")
        cmd.set("sphere_scale", 0.4, "DA_acc")

        _arrow("DA_arrow", a_best.coord, b_best.coord, rgb=(0.9,0.4,0.0))
        cmd.distance("DA_dist", "DA_don", "DA_acc")

    print(f"=== Predicted ET ({theory.upper()}) for H0 = {engine_params['H0']:.4f} eV ===")
    print(f"R_eff = {geom_dict['R_eff']:.2f} Å, f_orient = {f_orient:.3f}, Hab = {Hab*1000:.2f} meV, k = {k:.3e} s^-1, τ = {tau_ps:.2f} ps")

cmd.extend("et_rate_marcus", et_rate_marcus)

# ===== Marcus–Levich–Jortner =====

def et_rate_MLJ(*args, _self=cmd, **kwargs):
    """
    See USAGE_MLJ or run: et_help
    """

    geom_mode, args = _extract_mode_and_args(args)  # new line

    if len(args) < 5:
        print(USAGE_MLJ); return

    try:
        obj, don_chain, don_resi, acc_chain, acc_resi = [str(a).strip() for a in args[:5]]
        lambda_eV = float(_get_arg(args, 5, DEF_LAMBDA))
        dG    = float(_get_arg(args, 6, DEF_DELTAG))
        T     = float(_get_arg(args, 7, DEF_TK))
        H0v   = float(_get_arg(args, 8, DEF_H0))
        beta  = float(_get_arg(args, 9, DEF_BETA))
        R0v   = float(_get_arg(args,10, DEF_R0))

        # Accept possibly comma-separated lists for S, hw, vmax
        S_str = _get_arg(args, 11, "1.0")
        hw_str = _get_arg(args, 12, "0.18")
        vmax_str = _get_arg(args, 13, "10")

        orient_mode  = _get_arg(args, 14, "off")
        orient_floor = float(_get_arg(args,15, DEF_Floor))
        orient_pow   = float(_get_arg(args, 16, DEF_POW))
    except Exception as e:
        print(f"Argument parsing error: {e}\n{USAGE_MLJ}"); return

    # Parse lists
    S_list = [float(x) for x in str(S_str).split(",")]
    hw_list = [float(x) for x in str(hw_str).split(",")]
    vmax_list = [int(x) for x in str(vmax_str).split(",")]

    # Validate equal lengths for multi-mode
    if len(S_list) > 1:
        if not (len(S_list) == len(hw_list) == len(vmax_list)):
            print("Error: In multi-mode MLJ, S, hw_eV, and vmax lists must have the same length.")
            print(f"   Got lengths: S={len(S_list)}, hw={len(hw_list)}, vmax={len(vmax_list)}")
            return
    
    R_eff, a_best, b_best = _min_edge_distance_pymol(obj, (don_chain, don_resi), (acc_chain, acc_resi), geom_mode=geom_mode)
    if R_eff is None:
        return
    Hab_geom = _Hab_from_distance(R_eff, H0v, beta, R0v)
    f_orient = _edge_orient_factor_pymol(obj, (don_chain, don_resi), (acc_chain, acc_resi), geom_mode=geom_mode, orient_mode=orient_mode, orient_floor=orient_floor, orient_pow=orient_pow)
    Hab = Hab_geom * f_orient

    if len(S_list) > 1:
        # Multi-mode MLJ
        k = _mlj_rate_multi(Hab, lambda_eV, dG, T,
                            S_list, hw_list, vmax_list)
    else:
        # Single mode MLJ
        k = _mlj_rate(Hab, lambda_eV, dG, T,
                      S=S_list[0], hw_eV=hw_list[0], vmax=vmax_list[0])
    tau_ps = 1e12/k if k > 0 else float('inf')

    # Clean up any previous objects
    _safe_delete("DA_arrow"); _safe_delete("DA_dist")
    _safe_delete("DA_don"); _safe_delete("DA_acc")

    if a_best and b_best:
        # Create donor and acceptor pseudoatoms
        cmd.pseudoatom("DA_don", pos=a_best.coord)
        cmd.color("blue", "DA_don")
        cmd.show("spheres", "DA_don")
        cmd.set("sphere_scale", 0.4, "DA_don")

        cmd.pseudoatom("DA_acc", pos=b_best.coord)
        cmd.color("orange", "DA_acc")
        cmd.show("spheres", "DA_acc")
        cmd.set("sphere_scale", 0.4, "DA_acc")

        # Draw arrow and distance label
        _arrow("DA_arrow", a_best.coord, b_best.coord, rgb=(0.6,0.2,0.8))
        cmd.distance("DA_dist", "DA_don", "DA_acc")

    print(f"=== MLJ ET Prediction ===")
    print(f"R_eff = {R_eff:.2f} Å, Hab = {Hab*1000:.2f} meV")
    print(f"λ_s = {lambda_eV:.2f} eV, ΔG = {dG:+.2f} eV, T = {T:.2f} K")
    if len(S_list) > 1:
       print("Multi-mode MLJ")
       print(f"S list = {S_list}, hw list = {hw_list} eV, vmax list = {vmax_list}")
    else:
       print("Single-mode MLJ")
       print(f"S = {S_list[0]:.2f}, hw = {hw_list[0]:.3f} eV, vmax = {vmax_list[0]}")

    print(f"k = {k:.3e} s^-1, τ = {tau_ps:.2f} ps")

cmd.extend("et_rate_MLJ", et_rate_MLJ)


def et_detect_network_path(obj, max_distance=12.0, H0=DEF_H0, beta=DEF_BETA,
                           theory='marcus', engine_params=None,
                           donor=None, acceptor=None,
                           orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                           visualize=True):
    """
    Detect redox-active network, compute + highlight fastest path between donor and acceptor.
    Cleans up old network/path arrows before drawing new ones.
    """
    try:
        engine_class = _engine_map[theory.lower()]
    except KeyError:
        raise ValueError(f"Unknown theory '{theory}'. Available: {list(_engine_map.keys())}")
    engine = engine_class(engine_params)

    # === CLEANUP PREVIOUS VISUALS ===
    for name in cmd.get_names('objects'):
        if name.startswith("NET_") or name.startswith("NET_pA_") or name.startswith("NET_pB_") \
           or name.startswith("PATH_") or name.startswith("PATH_pA_") or name.startswith("PATH_pB_"):
            cmd.delete(name)

    # Step 1: detect nodes
    nodes = []
    seen = set()
    model = cmd.get_model(f"({obj}) and not elem H")
    for atom in model.atom:
        resn = atom.resn.strip().upper()
        if (resn in donacc_names_dict or resn in namesFAD) and (atom.chain, atom.resi) not in seen:
            seen.add((atom.chain, atom.resi))
            nodes.append((atom.chain, atom.resi, resn))

    edges = []
    G = nx.Graph()

    # Step 2: build all edges
    for i in range(len(nodes)):
        nodeA_id = (nodes[i][0], nodes[i][1])
        G.add_node(nodeA_id, resn=nodes[i][2])
        for j in range(i+1, len(nodes)):
            nodeB_id = (nodes[j][0], nodes[j][1])
            R, a1, a2 = _min_edge_distance_pymol(obj, nodeA_id, nodeB_id, geom_mode="default")
            if R and R <= max_distance:
                f_orient = _edge_orient_factor_pymol(obj, nodeA_id, nodeB_id, geom_mode="default",
                                               orient_mode=orient_mode,
                                               orient_floor=orient_floor,
                                               orient_pow=orient_pow)
                Hab = _Hab_from_distance(R,
                                         engine_params.get('H0', DEF_H0),
                                         engine_params.get('beta', DEF_BETA),
                                         engine_params.get('R0', DEF_R0)) * f_orient
                rate = engine.compute_rate(Hab, {'R_eff': R, 'Hab_eV': Hab})
                edges.append((nodes[i], nodes[j], R, Hab, rate, a1.coord, a2.coord))

    # Step 3: normalize weights
    max_rate_global = max([e[4] for e in edges if e[4] > 0], default=1.0)
    for e in edges:
        nA, nB, R, Hab, rate, coordA, coordB = e
        weight = -np.log(rate / max_rate_global) if rate > 0 else 1e6
        G.add_edge((nA[0], nA[1]), (nB[0], nB[1]),
                   distance=R, Hab=Hab, rate=rate, weight=weight,
                   coords=(coordA, coordB))

    # Step 4: visualize all edges
    if visualize:
        for e in edges:
            nA, nB, R, Hab, rate, coordA, coordB = e
            color = (1.0, 0.5, 0.0) if rate > 1e10 else (0.2, 0.6, 0.9)
            _arrow(f"NET_{nA[0]}{nA[1]}_{nB[0]}{nB[1]}", coordA, coordB, rgb=color)
            cmd.pseudoatom(f"NET_pA_{nA[0]}{nA[1]}_{nB[0]}{nB[1]}", pos=coordA)
            cmd.pseudoatom(f"NET_pB_{nA[0]}{nA[1]}_{nB[0]}{nB[1]}", pos=coordB)
            cmd.set("sphere_scale", 0.25, f"NET_pA_{nA[0]}{nA[1]}_{nB[0]}{nB[1]}")
            cmd.set("sphere_scale", 0.25, f"NET_pB_{nA[0]}{nA[1]}_{nB[0]}{nB[1]}")

    # Step 5: report network
    print("=== Redox Network Detection ===")
    print("Nodes:")
    for n in nodes:
        print(f"  {n[2]} {n[0]}/{n[1]}")
    print("Edges:")
    for e in edges:
        nA, nB, dist, Hab, rate, _, _ = e
        print(f"  {nA[2]} {nA[0]}/{nA[1]} ↔ {nB[2]} {nB[0]}/{nB[1]}: "
              f"R={dist:.2f} Å, Hab={Hab*1000:.2f} meV, k={rate:.2e} s^-1")

    # Step 6: fastest path if donor & acceptor specified
    if donor and acceptor:
        donor_id = tuple(donor.strip().split(":"))
        acceptor_id = tuple(acceptor.strip().split(":"))
        if donor_id in G.nodes and acceptor_id in G.nodes:
            try:
                fastest_path = nx.shortest_path(G, source=donor_id, target=acceptor_id, weight='weight')
                print(f"\nFastest path {donor} → {acceptor}:")
                for node in fastest_path:
                    resn = G.nodes[node]['resn']
                    print(f"  {resn} {node[0]}/{node[1]}")
                # Highlight fastest path in red
                if visualize:
                    for a, b in zip(fastest_path[:-1], fastest_path[1:]):
                        coordA, coordB = G[a][b]['coords']
                        _arrow(f"PATH_{a[0]}{a[1]}_{b[0]}{b[1]}", coordA, coordB, rgb=(1.0, 0.0, 0.0))
                        cmd.pseudoatom(f"PATH_pA_{a[0]}{a[1]}_{b[0]}{b[1]}", pos=coordA)
                        cmd.pseudoatom(f"PATH_pB_{a[0]}{a[1]}_{b[0]}{b[1]}", pos=coordB)
                        cmd.set("sphere_scale", 0.3, f"PATH_pA_{a[0]}{a[1]}_{b[0]}{b[1]}")
                        cmd.set("sphere_scale", 0.3, f"PATH_pB_{a[0]}{a[1]}_{b[0]}{b[1]}")
                # Bottleneck rate
                path_rates = [G[a][b]['rate'] for a, b in zip(fastest_path[:-1], fastest_path[1:])]
                bottleneck_rate = min(path_rates)
                print(f"Estimated bottleneck rate: {bottleneck_rate:.2e} s^-1\n")
            except nx.NetworkXNoPath:
                print(f"No path found between {donor} and {acceptor}")
        else:
            print("Donor/Acceptor not in detected network.")

cmd.extend("et_detect_network_path", et_detect_network_path)

# --- Static ET bridge wrappers (for GUI consistency) ---

def et_bridge_chain_const(geom_mode, obj, dChain, dResi, aChain, aResi,
                          bridges_str, lambda_eV=DEF_LAMBDA,
                          deltaG_eV=DEF_DELTAG, T_K=DEF_TK,
                          H0=DEF_H0, beta=DEF_BETA, R0=DEF_R0,
                          deltaB=DEF_DeltaB,
                          orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """Static bridge-mediated ET with constant ΔB for all bridges."""
    bridges = _parse_bridge_list(bridges_str)
    _run_chain(obj, (dChain, dResi), bridges, (aChain, aResi),
               lambda_eV, deltaG_eV, T_K,
               H0, beta, R0, [deltaB] * len(bridges),
               geom_mode=geom_mode,
               orient_mode=orient_mode,
               orient_floor=orient_floor, orient_pow=orient_pow)

cmd.extend("et_bridge_chain_const", et_bridge_chain_const)


def et_bridge_chain_pot(geom_mode, obj, dChain, dResi, aChain, aResi,
                        bridges_str, EoxD, EredA, Ereds_csv,
                        lambda_eV=DEF_LAMBDA, deltaG_eV=DEF_DELTAG, T_K=DEF_TK,
                        H0=DEF_H0, beta=DEF_BETA, R0=DEF_R0,
                        orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """Static bridge-mediated ET with ΔB from redox potentials."""
    bridges = _parse_bridge_list(bridges_str)
    Ereds = _parse_float_list(Ereds_csv, N_expected=len(bridges))
    deltaB_list = [_site_gap_from_potentials(EoxD, EredA, Eb) for Eb in Ereds]
    _run_chain(obj, (dChain, dResi), bridges, (aChain, aResi),
               lambda_eV, deltaG_eV, T_K,
               H0, beta, R0, deltaB_list,
               geom_mode=geom_mode,
               orient_mode=orient_mode,
               orient_floor=orient_floor, orient_pow=orient_pow)

cmd.extend("et_bridge_chain_pot", et_bridge_chain_pot)

def et_bridge_chain_gf(geom_mode, obj, dCh, dRs, aCh, aRs,
                       bridges_str, Emode, emode_args_csv,
                       lambda_eV=DEF_LAMBDA, dG=DEF_DELTAG, T=DEF_TK,
                       H0=DEF_H0, beta=DEF_BETA, R0=DEF_R0,
                       gamma=0.05, connect="chain",
                       orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    GF bridge wrapper (GUI-friendly):
    - emode_args_csv: comma/semicolon-separated numbers from one Tkinter field
    - Parses into correct core emode_args list
    """
    bridges = _parse_bridge_list(bridges_str)
    nums = _parse_float_list(emode_args_csv)

    Emode = str(Emode).lower()
    if Emode == "pot":
        if len(nums) < 2 + len(bridges):
            raise ValueError(f"POT mode: need EoxD, EredA, and {len(bridges)} bridge Ereds")
        EoxD = nums[0]
        EredA = nums[1]
        bridge_Ereds = nums[2:2+len(bridges)]
        emode_args_list = [EoxD, EredA, bridge_Ereds]
    elif Emode == "const":
        if len(nums) < 1 + len(bridges):
            raise ValueError(f"CONST mode: need Etun and {len(bridges)} ΔB values")
        Etun = nums[0]
        deltaBs = nums[1:1+len(bridges)]
        emode_args_list = [Etun] + deltaBs
    else:
        raise ValueError(f"Unknown GF submode '{Emode}'")

    # Call core
    k = et_bridge_chain_gf_core(static_distance_func, obj, dCh, dRs, aCh, aRs,
                                bridges, Emode, emode_args_list,
                                float(lambda_eV), float(dG), float(T),
                                float(H0), float(beta), float(R0),
                                float(gamma), connect,
                                geom_mode, orient_mode, float(orient_floor), float(orient_pow))
    tau_ps = 1e12/k if k > 0 else float('inf')
    print(f"GF ET rate: k = {k:.3e} s^-1, tau = {tau_ps:.2f} ps")

cmd.extend("et_bridge_chain_gf", et_bridge_chain_gf)

def et_bridge_chain_hop(geom_mode, obj, dCh, dRs, aCh, aRs,
                        bridges_str, Emode, emode_args_csv,
                        lambda_eV=DEF_LAMBDA, T=DEF_TK,
                        H0=DEF_H0, beta=DEF_BETA, R0=DEF_R0,
                        orient_mode="off", orient_floor=0.0, orient_pow=1.0):
    """
    HOP bridge wrapper (GUI-friendly):
    - emode_args_csv: comma/semicolon-separated numbers from one Tkinter field
    - Parses into correct core emode_args list
    """
    bridges = _parse_bridge_list(bridges_str)
    nums = _parse_float_list(emode_args_csv)

    Emode = str(Emode).lower()
    if Emode == "pot":
        if len(nums) < 2 + len(bridges):
            raise ValueError(f"POT mode: need EoxD, EredA, and {len(bridges)} bridge Ereds")
        EoxD = nums[0]
        EredA = nums[1]
        bridge_Ereds = nums[2:2+len(bridges)]
        emode_args_list = [EoxD, EredA, bridge_Ereds]
    elif Emode == "dg":
        emode_args_list = nums
    else:
        raise ValueError(f"Unknown HOP submode '{Emode}'")

    k = et_bridge_chain_hop_core(static_distance_func, obj, dCh, dRs, aCh, aRs,
                                 bridges, Emode, emode_args_list,
                                 float(lambda_eV), float(T),
                                 float(H0), float(beta), float(R0),
                                 geom_mode, orient_mode, float(orient_floor), float(orient_pow))
    tau_ps = 1e12/k if k > 0 else float('inf')
    print(f"HOP ET rate: k = {k:.3e} s^-1, tau = {tau_ps:.2f} ps")

cmd.extend("et_bridge_chain_hop", et_bridge_chain_hop)

def et_fit_step_params_hop_redox(obj, csv_file,
                                 fit_lambda_per_step=False,
                                 beta_bounds=(0.5, 1.5),
                                 H0_bounds=(1e-4, 0.5),
                                 lambda_bounds=(0.1, 1.5),
                                 R0=DEF_R0, T=DEF_TK,
                                 orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                                 geom_mode="default"):
    """
    Fit hopping ET parameters separately for each step row in CSV.
    CSV format per row:
        step_chain_list, step_resi_list, Eox_list, k_exp

    step_chain_list : 'A,B' for donor and acceptor chains
    step_resi_list  : '472,382' etc.
    Eox_list        : oxidation potentials donor, acceptor
    k_exp           : experimental ET rate in s^-1

    Returns:
      List of dictionaries with fitted parameters per step.
    """

    results = []

    with open(csv_file, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rtype = row['type'].strip().lower()
            ch_ids = [c.strip() for c in row['site_chain_list'].split(",")]
            resis = [r.strip() for r in row['site_resi_list'].split(",")]
            eox_vals = [float(e.strip()) for e in row['Eox_list'].split(",")]
            k_exp_val = float(row['k_exp'])

            nodeA = (ch_ids[0], resis[0])
            nodeB = (ch_ids[1], resis[1])

            # Get distance + orientation
            R, _, _ = _min_edge_distance_pymol(obj, nodeA, nodeB, geom_mode=geom_mode)
            if R is None:
                print(f"Step {nodeA}->{nodeB} distance not found.")
                continue

            f_orient = _edge_orient_factor_pymol(obj, nodeA, nodeB,
                                                 geom_mode=geom_mode,
                                                 orient_mode=orient_mode,
                                                 orient_floor=orient_floor,
                                                 orient_pow=orient_pow)

            dG_fwd = eox_vals[1] - eox_vals[0]

            # Objective function
            def log_error_step(params):
                if fit_lambda_per_step:
                    H0_fit, beta_fit, lam_fit = params
                else:
                    H0_fit, beta_fit = params
                    lam_fit = DEF_LAMBDA
                Hab = _Vpair(R, H0_fit, beta_fit, R0) * f_orient
                k_calc = _marcus_rate(Hab, lam_fit, dG_fwd, T)
                return abs(np.log10(k_calc) - np.log10(k_exp_val))

            # Initial guesses:
            p0 = [DEF_H0, DEF_BETA] + ([DEF_LAMBDA] if fit_lambda_per_step else [])
            bounds = [H0_bounds, beta_bounds] + ([lambda_bounds] if fit_lambda_per_step else [])

            res_fit = minimize(log_error_step, p0, bounds=bounds)
            if not res_fit.success:
                print(f"Step {nodeA}->{nodeB} fit failed: {res_fit.message}")
                continue

            # Extract results
            if fit_lambda_per_step:
                H0_opt, beta_opt, lam_opt = res_fit.x
            else:
                H0_opt, beta_opt = res_fit.x
                lam_opt = DEF_LAMBDA

            Hab_opt = _Vpair(R, H0_opt, beta_opt, R0) * f_orient
            k_calc_opt = _marcus_rate(Hab_opt, lam_opt, dG_fwd, T)
            log_err = np.log10(k_calc_opt) - np.log10(k_exp_val)

            results.append({
                'donor': nodeA,
                'acceptor': nodeB,
                'R': R,
                'f_orient': f_orient,
                'H0': H0_opt,
                'beta': beta_opt,
                'lambda': lam_opt,
                'dG_fwd': dG_fwd,
                'Hab_meV': Hab_opt * 1000,
                'k_exp': k_exp_val,
                'k_calc': k_calc_opt,
                'log_err': log_err
            })

            print(f"[Step] {nodeA[0]}:{nodeA[1]} -> {nodeB[0]}:{nodeB[1]}:"
                  f" R={R:.2f} Å, f_orient={f_orient:.3f}, "
                  f"H0={H0_opt:.5f} eV, β={beta_opt:.3f}, λ={lam_opt:.3f} eV, "
                  f"Hab={Hab_opt*1000:.2f} meV, k_exp={k_exp_val:.3e}, k_calc={k_calc_opt:.3e}, log_err={log_err:+.3f}")

    return results

cmd.extend("et_fit_step_params_hop_redox", et_fit_step_params_hop_redox)

def et_fit_global_params_hop_redox_flex(obj, csv_file,
                                        fit_lambda=False,
                                        fix_beta=None,
                                        beta_bounds=None,
                                        H0_init=DEF_H0, beta_init=DEF_BETA, lam_init=DEF_LAMBDA,
                                        R0=DEF_R0, T=DEF_TK,
                                        orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                                        geom_mode="default"):
    """
    Flexible hole-hopping fitter with beta bounds.

    Parameters:
      fit_lambda : bool
         If True, fit λ along with H0 and β (or H0 if β is fixed).
      fix_beta : None or float
         If float, β is fixed to this value and not fitted.
      beta_bounds : None or (min,max) tuple
         Bounds for β during fit (ignored if fix_beta is not None).
      H0_init, beta_init, lam_init : float
         Initial guesses.
      R0, T : float
         Reference distance (Å) and temperature (K).
      orient_mode, orient_floor, orient_pow, geom_mode : geometry/orientation settings.

    CSV format:
      type,site_chain_list,site_resi_list,Eox_list,k_exp
        type: 'overall' (full chain MFPT) or 'step' (single hop).
        site_chain_list: comma-separated chain IDs in HOLE hopping order.
        site_resi_list: comma-separated residue numbers.
        Eox_list: oxidation potentials (V vs NHE).
        k_exp: experimental rate (s^-1).
    """
    # ---- Helpers for CLI type safety ----
    def to_float_or_none(val):
        if isinstance(val, str):
            if val.strip().lower() in ["none", ""]:
                return None
            else:
                return float(val)
        elif val is None:
            return None
        else:
            return float(val)

    def to_tuple_or_none(val):
        if isinstance(val, str):
            txt = val.strip()
            if txt.lower() in ["none", ""]:
                return None
            # remove any surrounding parentheses
            txt = txt.strip("()")
            # split by comma
            parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
            if len(parts) != 2:
                raise ValueError(f"beta_bounds string must have two numbers, got: {val}")
            return (float(parts[0]), float(parts[1]))
        elif isinstance(val, (tuple, list)):
            if len(val) != 2:
                raise ValueError(f"beta_bounds tuple/list must have two numbers, got: {val}")
            return (float(val[0]), float(val[1]))
        elif val is None:
            return None
        else:
            raise ValueError(f"Cannot parse beta_bounds from: {val}")

    # ---- Convert inputs ----
    try:
        H0_init = float(H0_init)
        beta_init = float(beta_init)
        lam_init = float(lam_init)
        R0 = float(R0)
        T = float(T)
        orient_floor = float(orient_floor)
        orient_pow = float(orient_pow)
        fix_beta = to_float_or_none(fix_beta)
        beta_bounds = to_tuple_or_none(beta_bounds)
    except Exception as e:
        print("Parameter conversion error:", e)
        return

    if isinstance(fit_lambda, str):
        fit_lambda = fit_lambda.strip().lower() in ["true", "1", "yes"]

    # ---- Load CSV entries ----
    entries = []
    with open(csv_file, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rtype = row['type'].strip().lower()
            ch_ids = [c.strip() for c in row['site_chain_list'].split(",")]
            resis = [r.strip() for r in row['site_resi_list'].split(",")]
            eox_vals = [float(e.strip()) for e in row['Eox_list'].split(",")]
            k_exp_val = float(row['k_exp'])
            entries.append((rtype, ch_ids, resis, eox_vals, k_exp_val))

    # ---- Initial guess & bounds ----
    if fix_beta is None:
        p0 = [H0_init, beta_init]
        if beta_bounds is not None:
            bounds = [(1e-4, 0.5), (beta_bounds[0], beta_bounds[1])]
        else:
            bounds = [(1e-4, 0.5), (0.0, 3.0)]
        if fit_lambda:
            p0.append(lam_init)
            bounds.append((0.1, 1.5))
    else:
        p0 = [H0_init]
        bounds = [(1e-4, 0.5)]
        if fit_lambda:
            p0.append(lam_init)
            bounds.append((0.1, 1.5))

    # ---- Objective ----
    def log_error(params):
        if fix_beta is None:
            H0_fit = params[0]
            beta_fit = params[1]
            lam_fit = params[2] if fit_lambda else lam_init
        else:
            H0_fit = params[0]
            beta_fit = float(fix_beta)
            lam_fit = params[1] if fit_lambda else lam_init

        errs = []
        for rtype, ch_ids, resis, eox_vals, k_exp_val in entries:
            if rtype == "overall":
                V_edges, dG_steps = [], []
                for i in range(len(ch_ids)-1):
                    nodeA = (ch_ids[i], resis[i])
                    nodeB = (ch_ids[i+1], resis[i+1])
                    R, _, _ = _min_edge_distance_pymol(obj, nodeA, nodeB, geom_mode=geom_mode)
                    if R is None:
                        return np.inf
                    f_orient = _edge_orient_factor_pymol(obj, nodeA, nodeB,
                                                   geom_mode=geom_mode,
                                                   orient_mode=orient_mode,
                                                   orient_floor=orient_floor,
                                                   orient_pow=orient_pow)
                    V_edges.append(max(_Vpair(R, H0_fit, beta_fit, R0) * f_orient, 1e-12))
                    dG_steps.append(eox_vals[i+1] - eox_vals[i])
                try:
                    k_calc = _effective_rate_hopping_chain(V_edges, dG_steps, lam_fit, T)
                except np.linalg.LinAlgError:
                    return np.inf

            elif rtype == "step":
                nodeA = (ch_ids[0], resis[0])
                nodeB = (ch_ids[1], resis[1])
                R, _, _ = _min_edge_distance_pymol(obj, nodeA, nodeB, geom_mode=geom_mode)
                if R is None:
                    return np.inf
                f_orient = _edge_orient_factor_pymol(obj, nodeA, nodeB,
                                               geom_mode=geom_mode,
                                               orient_mode=orient_mode,
                                               orient_floor=orient_floor,
                                               orient_pow=orient_pow)
                Hab = max(_Vpair(R, H0_fit, beta_fit, R0) * f_orient, 1e-12)
                dG = eox_vals[1] - eox_vals[0]
                k_calc = _marcus_rate(Hab, lam_fit, dG, T)
            else:
                return np.inf

            if k_calc <= 0:
                return np.inf
            errs.append(np.log10(k_calc) - np.log10(k_exp_val))

        return np.sqrt(np.mean(np.square(errs)))

    # ---- Fit ----
    res = minimize(log_error, p0, method='L-BFGS-B', bounds=bounds)
    if not res.success:
        print(f"Fit failed: {res.message}")
        return

    if fix_beta is None:
        H0_opt = res.x[0]
        beta_opt = res.x[1]
        lam_opt = res.x[2] if fit_lambda else lam_init
    else:
        H0_opt = res.x[0]
        beta_opt = float(fix_beta)
        lam_opt = res.x[1] if fit_lambda else lam_init

    # ---- Reporting ----
    print("\n=== Fit Results (Overall + Step data) ===")
    print(f"Optimized H0   = {H0_opt:.5f} eV")
    print(f"β used         = {beta_opt:.5f} Å^-1 {'(fixed)' if fix_beta is not None else '(fitted)'}")
    if fit_lambda:
        print(f"Optimized λ    = {lam_opt:.5f} eV")
    print(f"Global RMS log10-error = {res.fun:.4f} (factor {10**res.fun:.2f})\n")

    for rtype, ch_ids, resis, eox_vals, k_exp_val in entries:
        if rtype == "overall":
            V_edges, dG_steps, R_list = [], [], []
            for i in range(len(ch_ids)-1):
                nodeA = (ch_ids[i], resis[i])
                nodeB = (ch_ids[i+1], resis[i+1])
                R, _, _ = _min_edge_distance_pymol(obj, nodeA, nodeB, geom_mode=geom_mode)
                f_orient = _edge_orient_factor_pymol(obj, nodeA, nodeB,
                                               geom_mode=geom_mode,
                                               orient_mode=orient_mode,
                                               orient_floor=orient_floor,
                                               orient_pow=orient_pow)
                Hab = max(_Vpair(R, H0_opt, beta_opt, R0) * f_orient, 1e-12)
                V_edges.append(Hab)
                R_list.append(R)
                dG_fwd = eox_vals[i+1] - eox_vals[i]
                dG_steps.append(dG_fwd)
            k_calc = _effective_rate_hopping_chain(V_edges, dG_steps, lam_opt, T)
            print(f"[Overall] {list(zip(ch_ids, resis))} => k_exp={k_exp_val:.3e}, k_calc={k_calc:.3e}")
            for i in range(len(ch_ids)-1):
                Hab = V_edges[i]
                dG_fwd = dG_steps[i]
                dG_bwd = -dG_fwd
                k_fwd = _marcus_rate(Hab, lam_opt, dG_fwd, T)
                k_bwd = _marcus_rate(Hab, lam_opt, dG_bwd, T)
                print(f"   {ch_ids[i]}:{resis[i]} -> {ch_ids[i+1]}:{resis[i+1]}:"
                      f" R={R_list[i]:.2f} Å, ΔG_fwd={dG_fwd:+.3f} eV, Hab={Hab*1000:.2f} meV,"
                      f" FET={k_fwd:.3e}, BET={k_bwd:.3e}")

        elif rtype == "step":
            nodeA = (ch_ids[0], resis[0])
            nodeB = (ch_ids[1], resis[1])
            R, _, _ = _min_edge_distance_pymol(obj, nodeA, nodeB, geom_mode=geom_mode)
            f_orient = _edge_orient_factor_pymol(obj, nodeA, nodeB,
                                           geom_mode=geom_mode,
                                           orient_mode=orient_mode,
                                           orient_floor=orient_floor,
                                           orient_pow=orient_pow)
            Hab = max(_Vpair(R, H0_opt, beta_opt, R0) * f_orient, 1e-12)
            dG_fwd = eox_vals[1] - eox_vals[0]
            dG_bwd = -dG_fwd
            k_calc = _marcus_rate(Hab, lam_opt, dG_fwd, T)
            k_fwd = k_calc
            k_bwd = _marcus_rate(Hab, lam_opt, dG_bwd, T)
            log_err = np.log10(k_calc) - np.log10(k_exp_val)
            print(f"[Step] {ch_ids[0]}:{resis[0]} -> {ch_ids[1]}:{resis[1]}:"
                  f" R={R:.2f} Å, ΔG_fwd={dG_fwd:+.3f} eV, Hab={Hab*1000:.2f} meV,"
                  f" FET={k_fwd:.3e}, BET={k_bwd:.3e},"
                  f" k_exp={k_exp_val:.3e}, k_calc={k_calc:.3e}, log_err={log_err:+.3f}")

cmd.extend("et_fit_global_params_hop_redox_flex", et_fit_global_params_hop_redox_flex)

#------------------------- Dynamic Section ------------------------
def _select_atoms_for_residue(universe, chain, resi,geom_mode='default',donacc_names_dict=donacc_names_dict):
    if chain:
       sel_str=f"chainid {chain} and "
    else:
       sel_str=""
    ag = universe.select_atoms(sel_str + f"resid {resi}")
    if ag.n_atoms == 0:
       return mda.AtomGroup([], u)
    resn = ag.resnames[0].upper()
    # Use donacc_names_dict logic for selection
    if geom_mode !='default' and resn in donacc_names_dict:
        names = donacc_names_dict[resn]
        sel_str = sel_str + f"resid {resi} and name {' '.join(names)} and not name H*"
    else:
        sel_str = sel_str = sel_str +f"resid {resi} and not name H* and not name N and not name CA and not name C and not name O"
    sel_atoms = universe.select_atoms(sel_str)
    return sel_atoms if sel_atoms.n_atoms > 0 else ag.select_atoms("not name H*")

def plot_histogram_with_params(data, bins, title_main,
                               engine_params, geom_mode, orient_mode, orient_floor, orient_pow,
                               extra_note=None):

    plt.figure()  # spawn a new figure/window

    plt.hist(data, bins=bins, alpha=0.7)
    plt.xlabel("ET rate (s^-1)")
    plt.ylabel("Frame count")

    # Metadata for reproducibility
    title_sub = (
        f"Geom: {geom_mode}, Orient: {orient_mode} "
        f"(floor={orient_floor}, pow={orient_pow})\n"
        f"λ={engine_params['lambda_eV']:.2f} eV, ΔG={engine_params['deltaG_eV']:+.2f} eV, T={engine_params['T_K']:.1f} K, "
        f"H0={engine_params['H0']:.3f} eV, β={engine_params['beta']:.2f} Å⁻¹, R0={engine_params['R0']:.2f} Å"
    )

    plt.title(f"{title_main}\n{title_sub}")

    # Annotate frame count
    plt.text(0.95, 0.95, f"{len(data)} frames",
             transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

    # Optional footer note
    if extra_note:
        plt.figtext(0.5, 0.01, extra_note, ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()
    return

def et_rate_marcus_MD(top_file, traj_file, donor_chain, donor_resi, acceptor_chain, acceptor_resi,
                      orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                      geom_mode="default", plot_hist=False, use_kubo=False,
                      theory="marcus", engine_params=None):
    """
    Computes average ET rate over an MD trajectory with optional Kubo–Anderson correction,
    supporting multiple theories via ETEngine.
    """

    # Instantiate engine based on theory
    try:
        engine_class = _engine_map[theory.lower()]
    except KeyError:
        raise ValueError(f"Unknown theory '{theory}'. Available: {list(_engine_map.keys())}")
    engine = engine_class(engine_params) 

    if theory == 'tsh':
        print("Warning: TSH for MD is computationally intensive (stochastic trajectories per frame). Consider reducing n_trajectories.")

    try:
        u = mda.Universe(top_file, traj_file)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return

    rates = []
    R_eff_vals = []
    Hab_vals = []

    donor_atoms = _select_atoms_for_residue(u, donor_chain, donor_resi,geom_mode=geom_mode)
    acc_atoms =   _select_atoms_for_residue(u, acceptor_chain, acceptor_resi,geom_mode=geom_mode)

    if donor_atoms.n_atoms == 0 or acc_atoms.n_atoms == 0:
        print("Error: Could not select donor or acceptor atoms.")
        return

    for ts in u.trajectory:
        coords_donor = donor_atoms.positions
        coords_acceptor = acc_atoms.positions

        k_frame, _, geom_dict, f_orient, Hab, _, _, _, _ = _compute_et_rate_unified(
            engine, coords_donor, coords_acceptor,
            orient_mode=orient_mode, orient_floor=orient_floor, orient_pow=orient_pow )

        rates.append(k_frame)
        R_eff_vals.append(geom_dict['R_eff'])
        Hab_vals.append(Hab)

    if not rates:
        print("No valid frames computed.")
        return

    # Compute averages
    k_avg = np.mean(rates)
    k_std = np.std(rates)
    R_avg = np.mean(R_eff_vals)
    Hab_avg = np.mean(Hab_vals)
    tau_ps = 1e12 / k_avg if k_avg > 0 else float('inf')

    print("=== Dynamic Disorder Averaged ET (MD) ===")
    print(f"Theory: {theory.upper()}")
    print(f"Frames analyzed: {len(rates)}")
    print(f"<R_eff> = {R_avg:.2f} Å")
    print(f"<Hab> = {Hab_avg*1000:.2f} meV")
    print(f"<k> = {k_avg:.3e} s^-1 ± {k_std:.3e} (std)")

    print(f"τ_avg = {tau_ps:.2f} ps")

    # Optional: Kubo–Anderson correlation correction
    tau_c = None

    if use_kubo:
        Hab_vals = np.array(Hab_vals)
        acf = _autocorr_fft(Hab_vals)
        times = np.arange(len(acf)) * u.trajectory.dt  # time lag in ps
        try:
            popt, _ = curve_fit(_exp_decay, times[:50], acf[:50], p0=[10.0])
            tau_c = popt[0]
            # Note: Formula kept as original; consider flipping for correct limits if needed
            k_eff = k_avg / (1 + (tau_c / tau_ps))
            tau_eff_ps = 1e12 / k_eff if k_eff > 0 else float('inf')
            print(f"\n--- Kubo–Anderson Correction (applies ~Hab^2 scaling) ---")
            print(f"τ_c (Hab corr) = {tau_c:.2f} ps")
            print(f"Kubo-corrected k_eff = {k_eff:.3e} s^-1, τ_eff = {tau_eff_ps:.2f} ps")
        except Exception as e:
            print(f"Kubo correlation fit failed: {e}")

        # Plot autocorrelation + fit
        try:
            plt.figure()
            plt.plot(times, acf, label="ACF (Hab)", marker="o", ms=3)
            if tau_c is not None:
                plt.plot(times, np.exp(-times / tau_c),
                         label=f"Fit exp(-t/{tau_c:.2f} ps)", color="r", lw=2)
            plt.xlabel("Lag time (ps)")
            plt.ylabel("Normalized ACF")
            plt.title("Hab autocorrelation from MD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Autocorrelation plot failed: {e}")
        # --- Spectral density plot ---
        try:
            dt_ps = u.trajectory.dt  # timestep in ps
            freqs_THz = rfftfreq(len(acf), d=dt_ps * 1e-12) * 1e-12  # THz from ps
            Jw = np.abs(rfft(acf))    # magnitude of FFT

            # Converted THz to cm⁻¹
            freqs_cm = freqs_THz * 33.356  # convert THz → cm⁻¹ (approx)
            plt.figure()
            plt.plot(freqs_cm, Jw, color="green", lw=1.5)
            plt.xlabel("Frequency (cm⁻¹)")
            plt.ylabel("|J(ω)| (a.u.)")
            plt.title("Spectral density of Hab fluctuations")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Spectral density plots failed: {e}")

    if plot_hist:
        extra_note = f"Theory: {theory.upper()}"
        if use_kubo and tau_c is not None:
            extra_note += f", Kubo τ_c={tau_c:.2f} ps"
        plot_histogram_with_params(
            rates, 50,
            "Per-frame ET rate distribution",
            engine_params, geom_mode, orient_mode, orient_floor, orient_pow,
            extra_note=extra_note
        )
    return

cmd.extend("et_rate_marcus_MD", et_rate_marcus_MD)

def et_bridge_chain_dynamic_MD(top_file, traj_file, dChain, dResi, bridges, aChain, aResi,
                               bridge_mode, Emode, emode_args_list,
                               orient_mode="off", orient_floor=0.0, orient_pow=1.0,
                               geom_mode="default", plot_hist=False, 
                               do_segment_analysis=False, segment_plot=False,
                               theory="marcus", engine_params=None):
    """
    Dynamic disorder averaged bridge-mediated ET over MD trajectory with unified geometry handling.
    Supports multiple theories via _engine_map.
    """

    # Instantiate engine via _engine_map
    try:
        engine_class = _engine_map[theory.lower()]
    except KeyError:
        raise ValueError(f"Unknown theory '{theory}'. Available: {list(_engine_map.keys())}")
    engine = engine_class(engine_params)

    if theory.lower() == 'tsh':
        print("Warning: TSH for bridge MD is computationally intensive.")

    try:
        u = mda.Universe(top_file, traj_file)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return

    k_list = []
    frames_ok = 0

    # Node list for bridge: donor + bridges + acceptor
    all_nodes = [(dChain, dResi)] + bridges + [(aChain, aResi)]
    segment_couplings_over_time = [[] for _ in range(len(all_nodes)-1)]
    segment_orient_over_time = [[] for _ in range(len(all_nodes)-1)]

    # Frame loop
    for ts in u.trajectory:
        coords_per_node = {}
        valid_frame = True

        # Get coords for all nodes in this frame
        for node in all_nodes:
            ch, rs = node
            sel_atoms = _select_atoms_for_residue(u, ch, int(rs), geom_mode=geom_mode)
            if sel_atoms.n_atoms == 0:
                valid_frame = False
                break
            coords_per_node[node] = sel_atoms.positions

        if not valid_frame:
            continue

        segment_info = []

        # Compute coupling for each segment using unified backend
        for i in range(len(all_nodes) - 1):
            nA = all_nodes[i]
            nB = all_nodes[i + 1]

            k_seg, _, geom_dict, f_orient, Hab, _, _, _, _ = _compute_et_rate_unified(
                engine,
                coords_per_node[nA],         # raw coords: donor side of segment
                coords_per_node[nB],         # raw coords: acceptor side of segment
                orient_mode=orient_mode,
                orient_floor=orient_floor,
                orient_pow=orient_pow
            )

            # Store segment R_eff, f_orient, Hab (coupling)
            segment_info.append((geom_dict['R_eff'], f_orient, Hab))
            segment_couplings_over_time[i].append(Hab if Hab is not None else 0.0)
            segment_orient_over_time[i].append(f_orient)

        # Compute Hab_eff depending on bridge_mode
        if bridge_mode in ("const", "pot"):
            if bridge_mode == "const":
                denom = emode_args_list[0] ** len(bridges)
            else:  # 'pot'
                EoxD, EredA = emode_args_list[0], emode_args_list[1]
                Ereds_br = emode_args_list[2:2 + len(bridges)]
                deltaB_list = [_site_gap_from_potentials(EoxD, EredA, Eb) for Eb in Ereds_br]
                denom = np.prod(deltaB_list)

            Hab_eff = np.prod([seg[2] for seg in segment_info]) / (denom if denom != 0 else 1e-30)
            k_frame = engine.compute_rate(Hab_eff)
        else:  # gf/hop fallback — default to Marcus internally
            if theory.lower() != 'marcus':
                print("Warning: GF/Hop modes currently use internal Marcus rates.")
            Hab_eff = np.prod([seg[2] for seg in segment_info])
            k_frame = _marcus_rate(Hab_eff, engine_params.get('lambda_eV', DEF_LAMBDA),
                                   engine_params.get('deltaG_eV', DEF_DELTAG),
                                   engine_params.get('T_K', DEF_TK))

        k_list.append(k_frame)
        frames_ok += 1

        # Reporting on first valid frame
        if frames_ok == 1:
            print(f"===== Dynamic {bridge_mode.upper()} bridge ET (MD, Theory: {theory.upper()}) =====")
            lab_nodes = []
            for ch, rs in all_nodes:
                ag = u.select_atoms(f"resid {rs}")
                resn = ag.resnames[0].upper() if ag.n_atoms > 0 else "UNK"
                lab_nodes.append(f"{resn} {ch}/{rs}")
            print("Path: " + "  ->  ".join(lab_nodes))
            print(f"Geometry mode: {geom_mode}, orient_mode={orient_mode}, orient_floor={orient_floor:.2f}, orient_pow={orient_pow:.2f}")
            print(f"Params: λ={engine_params['lambda_eV']:.2f} eV, "
                  f"ΔG={engine_params['deltaG_eV']:+.2f} eV, T={engine_params['T_K']:.2f} K, "
                  f"H0={engine_params['H0']:.3f} eV, β={engine_params['beta']:.2f} Å^-1, R0={engine_params['R0']:.2f} Å")
            print("Segments:")
            for i, seg in enumerate(segment_info):
                R_eff_val, f_orient_val, V_val = seg
                print(f"  {lab_nodes[i]} -> {lab_nodes[i+1]}: R_eff={R_eff_val:5.2f} Å, "
                      f"f_orient={f_orient_val:.3f}, V={V_val*1000:6.2f} meV")
            if bridge_mode in ("const", "pot"):
                print(f"Hab_eff = {Hab_eff*1000:.2f} meV")
            print(f"k (first frame) = {k_frame:.3e} s^-1, τ = {(1e12/k_frame) if k_frame>0 else float('inf'):.2f} ps")
            print("======================================================")

    if not k_list:
        print("No valid frames computed.")
        return

    # Averages
    k_avg = np.mean(k_list)
    k_std = np.std(k_list)
    print(f"Frames analysed: {frames_ok}")
    print(f"<k> = {k_avg:.3e} s^-1 ± {k_std:.3e} (std)")

    # Segment analysis
    if do_segment_analysis:
        n_segments = len(segment_couplings_over_time)
        print("\n=== Segment-resolved coupling analysis ===")
        for seg_idx in range(n_segments):
            V_array = np.array(segment_couplings_over_time[seg_idx])
            mean_V = np.mean(V_array)
            std_V = np.std(V_array)
            cv_V = std_V / mean_V if mean_V != 0 else np.nan
            print(f"Segment {seg_idx}: mean V = {mean_V*1000:.2f} meV, std = {std_V*1000:.2f} meV, CV = {cv_V:.3f}")

        # Correlation matrix
        V_matrix = np.column_stack(segment_couplings_over_time)
        corr_matrix = np.corrcoef(V_matrix.T)
        print("\nCorrelation matrix between segments (Pearson r):")
        print(corr_matrix)

        if segment_plot:
            fig, ax = plt.subplots()
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(n_segments))
            ax.set_yticks(range(n_segments))
            ax.set_xticklabels([f"Seg {i}" for i in range(n_segments)])
            ax.set_yticklabels([f"Seg {i}" for i in range(n_segments)])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_title("Segment Coupling Correlation Heatmap")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()

    if plot_hist:
        plot_histogram_with_params(
            k_list, 50,
            f"Dynamic {bridge_mode.upper()} bridge ET rate distribution ({theory.upper()})",
            engine_params, geom_mode, orient_mode, orient_floor, orient_pow)

cmd.extend("et_bridge_chain_dynamic_MD", et_bridge_chain_dynamic_MD)
 
#----------------------------- GUI Section ----------------------------

def open_et_plugin_gui():
    # Create hidden root
    root = tk.Tk()
    root.withdraw()

    # Main plugin window
    win = tk.Toplevel(root)
    win.title("Electron Transfer Toolkit")
    win.geometry("900x800")

    # Close event
    def on_close():
        try:
            root.destroy()
        except tk.TclError:
            pass
    win.protocol("WM_DELETE_WINDOW", on_close)

    # Notebook widget
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True)

    # Output text box with stdout redirect
    result_box = tk.Text(win, height=60, wrap="word")
    result_box.pack(fill="x", padx=5, pady=5)

    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget
        def write(self, s):
            self.widget.insert(tk.END, s)
            self.widget.see(tk.END)
        def flush(self):
            pass

    sys.stdout = TextRedirector(result_box)

    # Shared helpers
    def add_field(frame, label, row, var_type, default, width=14):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=3, pady=2)
        var = var_type(value=default)
        tk.Entry(frame, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=3, pady=2)
        return var

    def add_dropdown(frame, label, row, options, default):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=3, pady=2)
        var = tk.StringVar(value=default)
        ttk.Combobox(frame, textvariable=var, values=options, state="readonly", width=12).grid(row=row, column=1, sticky="w", padx=3, pady=2)
        return var

    def add_dropdown_widget(frame, label, row, options, default):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=3, pady=2)
        var = tk.StringVar(value=default)
        widget = ttk.Combobox(frame, textvariable=var, values=options, state="readonly", width=12)
        widget.grid(row=row, column=1, sticky="w", padx=3, pady=2)
        return var, widget

    def add_orient_params(frame, start_row):
        omode = add_dropdown(frame, "orient_mode", start_row, ["off", "on"], "off")
        ofloor = add_field(frame, "orient_floor", start_row+1, tk.DoubleVar, 0.0)
        opow = add_field(frame, "orient_pow", start_row+2, tk.DoubleVar, 1.0)
        return omode, ofloor, opow

    # Call modular builders for each tab
    _build_tab_detect_network(nb, add_field, add_dropdown, add_orient_params)
    _build_tab_calibrate(nb, add_field, add_dropdown, add_orient_params)
    _build_tab_rate(nb, add_field, add_dropdown, add_orient_params)
    _build_tab_static_bridge(nb, add_field, add_dropdown, add_dropdown_widget, add_orient_params)
    _build_tab_fit_hop(nb, add_field, add_dropdown, add_orient_params)
    _build_tab_marcus_md(nb, add_field, add_dropdown, add_orient_params)
    _build_tab_dynamic_bridge_md(nb, add_field, add_dropdown, add_dropdown_widget, add_orient_params)
    _build_tab_aromatic_metrics(nb, add_field, add_dropdown)

    # Start GUI loop
    root.mainloop()

# Register
cmd.extend("et_plugin_gui", lambda: open_et_plugin_gui())

def build_engine_params_generic(vars_dict, theory):
    # Base defaults
    params = {
        'lambda_eV': float(vars_dict['lambda_eV'].get()) if vars_dict['lambda_eV'].get() != "" else DEF_LAMBDA,
        'deltaG_eV': float(vars_dict['deltaG_eV'].get()) if vars_dict['deltaG_eV'].get() != "" else DEF_DELTAG,
        'T_K':       float(vars_dict['T_K'].get())       if vars_dict['T_K'].get()       != "" else DEF_TK,
        'H0':        float(vars_dict['H0'].get())        if vars_dict['H0'].get()        != "" else DEF_H0,
        'beta':      float(vars_dict['beta'].get())      if vars_dict['beta'].get()      != "" else DEF_BETA,
        'R0':        float(vars_dict['R0'].get())        if vars_dict['R0'].get()        != "" else DEF_R0
    }

    th = theory.lower()

    # Theory-specific additions
    if th == 'mlj':
        params['lambda_s_eV'] = params['lambda_eV']
        params.update({
            'S_list': _parse_float_list(vars_dict['S_list'].get()) or [1.0],
            'hw_list': _parse_float_list(vars_dict['hw_list'].get()) or [0.18],
            'vmax_list': [int(v) for v in _parse_float_list(vars_dict['vmax_list'].get())] or [10]
        })
    elif th == 'redfield':
        params['gamma_deph_s1'] = float(vars_dict['gamma_deph_s1'].get()) if vars_dict['gamma_deph_s1'].get() != "" else 5e13
    elif th == 'tsh':
        params.update({
            'mass_amu': float(vars_dict['mass_amu'].get()) if vars_dict['mass_amu'].get() != "" else 100.0,
            'omega_cm': float(vars_dict['omega_cm'].get()) if vars_dict['omega_cm'].get() != "" else 1000.0,
            'n_trajectories': int(vars_dict['n_trajectories'].get()) if vars_dict['n_trajectories'].get() != "" else 100
        })

    return params

def add_csv_field(frame, label, row):
    """
    Create a labeled text field for a file path, plus a 'Browse' button.
    Returns the associated StringVar so you can call .get().
    """
    var = tk.StringVar(value="")

    # Label
    tk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=3, pady=2)

    # Entry box
    tk.Entry(frame, textvariable=var, width=25).grid(row=row, column=1, sticky="w", padx=3, pady=2)

    # Browse button
    def browse_file():
        fname = filedialog.askopenfilename(
            title="Select file",
            filetypes=[
                ("All files", "*.*"),
                ("CSV files", "*.csv"),
                ("Topology files", "*.prmtop"),
                ("Trajectory files", "*.nc")
            ]
        )
        if fname:
            var.set(fname)

    tk.Button(frame, text="Browse", command=browse_file).grid(row=row, column=2, padx=3, pady=2)
    return var

def create_collapsible_frame(parent, title, row_start):
    """Collapsible frame with checkbox toggle."""
    frame = ttk.LabelFrame(parent, text=title)
    frame.grid(row=row_start, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    
    # Toggle checkbox
    toggle_var = tk.BooleanVar(value=False)
    toggle_cb = ttk.Checkbutton(frame, text="Show Details", variable=toggle_var, command=lambda: frame.toggle())
    toggle_cb.grid(row=0, column=0, sticky="w")
    
    # Bind toggle
    def toggle():
        if toggle_var.get():
            for child in frame.winfo_children()[1:]:  # Skip checkbox
                child.grid()
        else:
            for child in frame.winfo_children()[1:]:
                child.grid_remove()
    
    frame.toggle = toggle  # Method attach
    toggle()  # Initially hidden
    return frame, toggle_var, row_start + 1  # Next row

def link_bridge_and_emode(bridge_mode_var, emode_var, emode_widget):
    """
    Adjusts Energy Submode dropdown values based on Bridge Mode selection.
    """
    def _update(*args):
        bmode = bridge_mode_var.get().lower()
        if bmode == "gf":
            emode_widget["values"] = ["pot", "const"]
            if emode_var.get() not in ("pot", "const"):
                emode_var.set("pot")
        elif bmode == "hop":
            emode_widget["values"] = ["pot", "dg"]
            if emode_var.get() not in ("pot", "dg"):
                emode_var.set("pot")
        elif bmode == "const":
            emode_widget["values"] = ["const"]
            emode_var.set("const")
        elif bmode == "pot":
            emode_widget["values"] = ["pot"]
            emode_var.set("pot")

    bridge_mode_var.trace_add("write", _update)

# GUI functions
# Tab 1: Detect ET Network
def _build_tab_detect_network(nb, add_field, add_dropdown, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Detect ET Network")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    dist = add_field(tab, "Max Distance (Å)", 1, tk.DoubleVar, 12.0)
    theory_var = add_dropdown(tab, "Theory", 2, list(_engine_map.keys()), "marcus")
    H0v = add_field(tab, "H0 (eV)", 3, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β (Å^-1)", 4, tk.DoubleVar, DEF_BETA)
    R0v = add_field(tab, "R0 (Å)", 5, tk.DoubleVar, DEF_R0)
    lambda_eV = add_field(tab, "λ (eV)", 6, tk.DoubleVar, DEF_LAMBDA)
    dG = add_field(tab, "ΔG (eV)", 7, tk.DoubleVar, DEF_DELTAG)
    T = add_field(tab, "T (K)", 8, tk.DoubleVar, DEF_TK)

    donor = add_field(tab, "Donor (Chain:Resi)", 9, tk.StringVar, "A:382")
    acceptor = add_field(tab, "Acceptor (Chain:Resi)", 10, tk.StringVar, "A:472")
    om, of, op = add_orient_params(tab, 11)
    vis = add_dropdown(tab, "Visualize", 14, ["True", "False"], "True")

    # Theory-specific frames 
    mlj_frame, mlj_toggle, next_row = create_collapsible_frame(tab, "MLJ-Specific Params", 15)
    S_var = add_field(mlj_frame, "S (CSV)", 0, tk.StringVar, "1.0")  
    hw_var = add_field(mlj_frame, "hw (CSV)", 1, tk.StringVar, "0.18")
    vmax_var = add_field(mlj_frame, "vmax (CSV)", 2, tk.StringVar, "10")
    mlj_frame.grid(row=15, column=0, columnspan=2, sticky="ew", padx=10, pady=5)  # Explicit grid
    mlj_frame.toggle()  # Hide initially

    red_frame, red_toggle, next_row = create_collapsible_frame(tab, "Redfield-Specific Params", 16)
    gamma_var = add_field(red_frame, "γ_deph (s^-1)", 0, tk.DoubleVar, 5e13)
    red_frame.grid(row=16, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    red_frame.toggle()

    tsh_frame, tsh_toggle, next_row = create_collapsible_frame(tab, "TSH-Specific Params", 17)
    mass_var = add_field(tsh_frame, "mass_amu", 0, tk.DoubleVar, 100.0)
    omega_var = add_field(tsh_frame, "omega_cm", 1, tk.DoubleVar, 1000.0)
    ntraj_var = add_field(tsh_frame, "n_trajectories", 2, tk.IntVar, 100)
    tsh_frame.grid(row=17, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    tsh_frame.toggle()

    def run_et_detect():
        # Build engine_params
        params_tk = {
            'lambda_eV': lambda_eV, 'deltaG_eV': dG, 'T_K': T,
            'H0': H0v, 'beta': betav, 'R0': R0v,
            'S_list': S_var,
            'hw_list': hw_var,'vmax_list': vmax_var,
            'gamma_deph_s1': gamma_var,
            'mass_amu': mass_var, 'omega_cm': omega_var, 'n_trajectories': ntraj_var }
        
        # Call extended et_rate_marcus (assumes updated to handle theory/params)
        et_detect_network_path(
            obj.get(), dist.get(),
            theory=theory_var.get(), engine_params=build_engine_params_generic(params_tk, theory_var.get()),
            donor=donor.get(), acceptor=acceptor.get(),
            orient_mode=om.get(), orient_floor=of.get(), orient_pow=op.get(),
            visualize=vis.get() )

    tk.Button(tab, text="Run Detection+Path",
              command=run_et_detect ).grid(row=29, column=0, columnspan=2, pady=5)

# Tab 2: Calibrate / Fit Params
def _build_tab_calibrate(nb, add_field, add_dropdown, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Calibrate H0/λ/β to k")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    dch = add_field(tab, "Donor Chain", 1, tk.StringVar, "A")
    drs = add_field(tab, "Donor Resi", 2, tk.StringVar, "382")
    ach = add_field(tab, "Acceptor Chain", 3, tk.StringVar, "A")
    ars = add_field(tab, "Acceptor Resi", 4, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 5, ["default", "selec_atoms"], "default")

    # New: Theory selection
    theory_var = add_dropdown(tab, "Theory", 6, list(_engine_map.keys()), "marcus")
    
    # Common params
    lambda_eV = add_field(tab, "λ init (eV)", 7, tk.DoubleVar, DEF_LAMBDA)
    H0v = add_field(tab, "H0 init (eV)", 8, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β init (Å⁻¹)", 9, tk.DoubleVar, DEF_BETA)
    dG = add_field(tab, "ΔG (eV)", 10, tk.DoubleVar, DEF_DELTAG)
    T = add_field(tab, "T [K]", 11, tk.DoubleVar, DEF_TK)
    R0v = add_field(tab, "R0 (Å)", 12, tk.DoubleVar, DEF_R0)
    ktar = add_field(tab, "Target k (s⁻¹)", 13, tk.DoubleVar, 1.25e12)

    # Theory-specific frames 
    mlj_frame, mlj_toggle, next_row = create_collapsible_frame(tab, "MLJ-Specific Params", 14)
    S_var = add_field(mlj_frame, "S (CSV)", 0, tk.StringVar, "1.0")  
    hw_var = add_field(mlj_frame, "hw (CSV)", 1, tk.StringVar, "0.18")
    vmax_var = add_field(mlj_frame, "vmax (CSV)", 2, tk.StringVar, "10")
    mlj_frame.grid(row=14, column=0, columnspan=2, sticky="ew", padx=10, pady=5)  # Explicit grid
    mlj_frame.toggle()  # Hide initially

    red_frame, red_toggle, next_row = create_collapsible_frame(tab, "Redfield-Specific Params", 15)
    gamma_var = add_field(red_frame, "γ_deph (s^-1)", 0, tk.DoubleVar, 5e13)
    red_frame.grid(row=15, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    red_frame.toggle()

    tsh_frame, tsh_toggle, next_row = create_collapsible_frame(tab, "TSH-Specific Params", 16)
    mass_var = add_field(tsh_frame, "mass_amu", 0, tk.DoubleVar, 100.0)
    omega_var = add_field(tsh_frame, "omega_cm", 1, tk.DoubleVar, 1000.0)
    ntraj_var = add_field(tsh_frame, "n_trajectories", 2, tk.IntVar, 100)
    tsh_frame.grid(row=16, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    tsh_frame.toggle()

    fit_params_opt = add_dropdown(tab, "Fit Params", 17,
                                  ["H0", "lambda", "beta",
                                   "H0+lambda", "H0+beta", "lambda+beta",
                                   "H0+lambda+beta"], "H0")

    plot_opt = add_dropdown(tab, "Plot", 18, ["None", "Yes"], "None")
    om, of, op = add_orient_params(tab, 19)

    vis_opt = add_dropdown(tab, "Visualize in PyMOL", 22, ["off", "on"], "off")

    # Fit ranges (common)
    fit_range_H0_min = add_field(tab, "Fit range H0 min", 23, tk.DoubleVar, 0.001)
    fit_range_H0_max = add_field(tab, "Fit range H0 max", 24, tk.DoubleVar, 0.5)
    fit_range_lambda_min = add_field(tab, "Fit range λ min", 25, tk.DoubleVar, 0.1)
    fit_range_lambda_max = add_field(tab, "Fit range λ max", 26, tk.DoubleVar, 2.0)
    fit_range_beta_min = add_field(tab, "Fit range β min", 27, tk.DoubleVar, 0.1)
    fit_range_beta_max = add_field(tab, "Fit range β max", 28, tk.DoubleVar, 2.0)

    # Callback to show/hide theory-specific frames
    def update_theory_fields(*args):
        th = theory_var.get().lower()
        mlj_frame.grid_remove()
        red_frame.grid_remove()
        tsh_frame.grid_remove()
        if th == 'mlj':
            mlj_frame.grid()
        elif th == 'redfield':
            red_frame.grid()
        elif th == 'tsh':
            tsh_frame.grid()
        theory_var.trace_add("write", update_theory_fields)
        update_theory_fields()  # Initial call
        return

    def run_calibrate():
        # Convert fit_params
        fsel = fit_params_opt.get()
        fit_tuple = tuple(fsel.split("+")) if "+" in fsel else (fsel,)
        
        # Build engine_params
        params_tk = {
            'lambda_eV': lambda_eV, 'deltaG_eV': dG, 'T_K': T,
            'H0': H0v, 'beta': betav, 'R0': R0v,
            'S_list': S_var,
            'hw_list': hw_var,'vmax_list': vmax_var,
            'gamma_deph_s1': gamma_var,
            'mass_amu': mass_var, 'omega_cm': omega_var, 'n_trajectories': ntraj_var }
        
        # Bounds
        bounds_dict = {
            'H0': (fit_range_H0_min.get(), fit_range_H0_max.get()),
            'lambda_eV': (fit_range_lambda_min.get(), fit_range_lambda_max.get()),
            'beta': (fit_range_beta_min.get(), fit_range_beta_max.get())
        }
 
        # Define sweep ranges (matching function defaults)
        sweep_range_H0 = (0.01, 0.2)
        sweep_range_lambda = (0.3, 1.2)
        sweep_range_beta = (0.1, 2.0)
       
        calibrate_Marcus_to_k(
            obj.get(), dch.get(), drs.get(), ach.get(), ars.get(),
            theory=theory_var.get(),
            engine_params=build_engine_params_generic(params_tk, theory_var.get()),
            k_target=ktar.get(),
            fit_params=fit_tuple,
            plot=None if plot_opt.get() == "None" else plot_opt.get(),
            sweep_range_H0=sweep_range_H0,  # Global, or per-theory if needed
            sweep_range_lambda=sweep_range_lambda,
            sweep_range_beta=sweep_range_beta,
            fit_range_H0=bounds_dict['H0'],
            fit_range_lambda=bounds_dict['lambda_eV'],
            fit_range_beta=bounds_dict['beta'],
            geom_mode=geom_mode.get(),
            orient_mode=om.get(),
            orient_floor=of.get(),
            orient_pow=op.get(),
            visualize=vis_opt.get()
        )

    tk.Button(tab, text="Run Calibration", command=run_calibrate).grid(row=29, column=0, columnspan=2, pady=5)

# TAB 3: ET Rate Calculator ----------------------
def _build_tab_rate(nb, add_field, add_dropdown, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="ET Rate Calc.")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    dch = add_field(tab, "Donor Chain", 1, tk.StringVar, "A")
    drs = add_field(tab, "Donor Resi", 2, tk.StringVar, "382")
    ach = add_field(tab, "Acceptor Chain", 3, tk.StringVar, "A")
    ars = add_field(tab, "Acceptor Resi", 4, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 5, ["default", "selec_atoms"], "default")
    
    # New: Theory selection
    theory_var = add_dropdown(tab, "Theory", 6, list(_engine_map.keys()), "marcus")
    
    # Common params
    lambda_eV = add_field(tab, "λ", 7, tk.DoubleVar, DEF_LAMBDA)
    dG = add_field(tab, "ΔG", 8, tk.DoubleVar, DEF_DELTAG)
    T = add_field(tab, "T [K]", 9, tk.DoubleVar, DEF_TK)
    H0v = add_field(tab, "H0", 10, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β", 11, tk.DoubleVar, DEF_BETA)
    R0v = add_field(tab, "R0", 12, tk.DoubleVar, DEF_R0)
    
    # Theory-specific frames (similar to calibrate)
    mlj_frame, mlj_toggle, next_row = create_collapsible_frame(tab, "MLJ-Specific Params", 13)
    S_var = add_field(mlj_frame, "S (CSV)", 0, tk.StringVar, "1.0")  
    hw_var = add_field(mlj_frame, "hw (CSV)", 1, tk.StringVar, "0.18")
    vmax_var = add_field(mlj_frame, "vmax (CSV)", 2, tk.StringVar, "10")
    mlj_frame.grid(row=13, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    mlj_frame.toggle()

    red_frame, red_toggle, next_row = create_collapsible_frame(tab, "Redfield-Specific Params", 14)
    gamma_var = add_field(red_frame, "γ_deph (s^-1)", 0, tk.DoubleVar, 5e13)
    red_frame.grid(row=14, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    red_frame.toggle()

    tsh_frame, tsh_toggle, next_row = create_collapsible_frame(tab, "TSH-Specific Params", 15)
    mass_var = add_field(tsh_frame, "mass_amu", 0, tk.DoubleVar, 100.0)
    omega_var = add_field(tsh_frame, "omega_cm", 1, tk.DoubleVar, 1000.0)
    ntraj_var = add_field(tsh_frame, "n_trajectories", 2, tk.IntVar, 100)
    tsh_frame.grid(row=15, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
    tsh_frame.toggle()

    om, of, op = add_orient_params(tab, 16)

    # Callback for fields
    def update_theory_fields(*args):
        th = theory_var.get().lower()
        mlj_frame.grid_remove()
        red_frame.grid_remove()
        tsh_frame.grid_remove()
        if th == 'mlj':
            mlj_frame.grid()
        elif th == 'redfield':
            red_frame.grid()
        elif th == 'tsh':
            tsh_frame.grid()
    theory_var.trace_add("write", update_theory_fields)
    update_theory_fields()

    def run_et_rate():
        # Build engine_params
        params_tk = {
            'lambda_eV': lambda_eV, 'deltaG_eV': dG, 'T_K': T,
            'H0': H0v, 'beta': betav, 'R0': R0v,
            'S_list': S_var,
            'hw_list': hw_var,'vmax_list': vmax_var,
            'gamma_deph_s1': gamma_var,
            'mass_amu': mass_var, 'omega_cm': omega_var, 'n_trajectories': ntraj_var }
        
        # Call extended et_rate_marcus (assumes updated to handle theory/params)
        et_rate_marcus(
            geom_mode.get(), obj.get(), dch.get(), drs.get(),
            ach.get(), ars.get(), build_engine_params_generic(params_tk, theory_var.get()),
            om.get(), of.get(), op.get(), theory=theory_var.get() )

    tk.Button(tab, text="Run", command=run_et_rate).grid(row=19, column=0, columnspan=2, pady=5)

# Tab 4: Static Bridge ET
def _build_tab_static_bridge(nb, add_field, add_dropdown, add_dropdown_widget, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Static Bridge ET")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    dch = add_field(tab, "Donor Chain", 1, tk.StringVar, "A")
    drs = add_field(tab, "Donor Resi", 2, tk.StringVar, "306")
    ach = add_field(tab, "Acceptor Chain", 3, tk.StringVar, "A")
    ars = add_field(tab, "Acceptor Resi", 4, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 5, ["default", "selec_atoms"], "default")
    calc_mode, calc_widget = add_dropdown_widget(tab, "Bridge Calc. Mode", 6, ["const", "pot", "gf", "hop"], "const")
    bridges = add_field(tab, "Bridges", 7, tk.StringVar, "")
    emode, emode_widget = add_dropdown_widget(tab, "Energy Submode", 8, ["pot", "const"], "pot")
    emode_args = add_field(tab, "Energy Args (CSV)", 9, tk.StringVar, "")
    lambda_eV = add_field(tab, "λ", 10, tk.DoubleVar, DEF_LAMBDA)
    dGv = add_field(tab, "ΔG", 11, tk.DoubleVar, DEF_DELTAG)
    Tv = add_field(tab, "T [K]", 12, tk.DoubleVar, DEF_TK)
    H0v = add_field(tab, "H0", 13, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β", 14, tk.DoubleVar, DEF_BETA)
    R0v = add_field(tab, "R0", 15, tk.DoubleVar, DEF_R0)
    om, of, op = add_orient_params(tab, 16)

    link_bridge_and_emode(calc_mode, emode, emode_widget)

    def run_static_bridge():
        cmode = calc_mode.get().lower()
        ea_csv = emode_args.get()
        # Handle bridge call as in original
        if cmode == "const":
            et_bridge_chain_const(
                geom_mode.get(), obj.get(), dch.get(), drs.get(), ach.get(), ars.get(),
                bridges.get(), lambda_eV.get(), dGv.get(), Tv.get(),
                H0v.get(), betav.get(), R0v.get(),
                float(ea_csv) if ea_csv else DEF_DeltaB,
                om.get(), of.get(), op.get()
            )
        elif cmode == "pot":
            parts = _parse_float_list(ea_csv)
            EoxD = parts[0]; EredA = parts[1]
            Ereds_csv = ",".join(str(v) for v in parts[2:])
            et_bridge_chain_pot(
                geom_mode.get(), obj.get(), dch.get(), drs.get(),
                ach.get(), ars.get(), bridges.get(),
                EoxD, EredA, Ereds_csv,
                lambda_eV.get(), dGv.get(), Tv.get(),
                H0v.get(), betav.get(), R0v.get(),
                om.get(), of.get(), op.get()
            )
        elif cmode == "gf":
            et_bridge_chain_gf(
                geom_mode.get(), obj.get(), dch.get(), drs.get(), ach.get(), ars.get(),
                bridges.get(), emode.get(), ea_csv,
                lambda_eV.get(), dGv.get(), Tv.get(),
                H0v.get(), betav.get(), R0v.get(), 0.05, "chain",
                om.get(), of.get(), op.get()
            )
        elif cmode == "hop":
            et_bridge_chain_hop(
                geom_mode.get(), obj.get(), dch.get(), drs.get(), ach.get(), ars.get(),
                bridges.get(), emode.get(), ea_csv,
                lambda_eV.get(), Tv.get(),
                H0v.get(), betav.get(), R0v.get(),
                om.get(), of.get(), op.get()
            )
        else:
            print(f"Unknown calc mode '{cmode}'")

    tk.Button(tab, text="Run", command=run_static_bridge).grid(row=19, column=0, columnspan=2, pady=5)


# Tab 9: Fit Hop Redox Flex

def _build_tab_fit_hop(nb, add_field, add_dropdown, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Fit hopping ET rate")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    csv_path = add_field(tab, "CSV File", 1, tk.StringVar, "")

    def browse_csv():
        fname = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if fname:
            csv_path.set(fname)
    tk.Button(tab, text="Browse", command=browse_csv).grid(row=1, column=2, padx=3, pady=2)

    fit_mode = add_dropdown(tab, "Fit Mode", 2, ["Global", "Per-step"], "Global")

    fit_lambda_var = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Fit λ", variable=fit_lambda_var).grid(row=3, column=1, sticky="w")

    fix_beta = add_field(tab, "Fix β (Global mode)", 4, tk.StringVar, "")
    beta_bounds = add_field(tab, "β bounds (min,max)", 5, tk.StringVar, "0.5,1.5")
    H0_init = add_field(tab, "H0_init", 6, tk.DoubleVar, DEF_H0)
    beta_init = add_field(tab, "β_init", 7, tk.DoubleVar, DEF_BETA)
    lam_init = add_field(tab, "λ_init", 8, tk.DoubleVar, DEF_LAMBDA)
    R0v = add_field(tab, "R0", 9, tk.DoubleVar, DEF_R0)
    Tv = add_field(tab, "T (K)", 10, tk.DoubleVar, DEF_TK)
    om, of, op = add_orient_params(tab, 11)
    geom_mode = add_dropdown(tab, "Geometry Mode", 14, ["default", "selec_atoms"], "default")

    tk.Button(
        tab, text="Run Fit",
        command=lambda: run_fit(
            mode=fit_mode.get(),
            obj=obj.get(),
            csv_path=csv_path.get(),
            fit_lambda=fit_lambda_var.get(),
            fix_beta=fix_beta.get(),
            beta_bounds=beta_bounds.get(),
            H0_init=H0_init.get(), beta_init=beta_init.get(), lam_init=lam_init.get(),
            R0=R0v.get(), T=Tv.get(),
            orient_mode=om.get(), orient_floor=of.get(), orient_pow=op.get(),
            geom_mode=geom_mode.get()
        )
    ).grid(row=15, column=0, columnspan=2, pady=5)

def run_fit(mode, obj, csv_path, fit_lambda, fix_beta, beta_bounds,
            H0_init, beta_init, lam_init, R0, T, orient_mode, orient_floor, orient_pow, geom_mode):
    if mode.lower() == "global":
        et_fit_global_params_hop_redox_flex(
            obj, csv_path, fit_lambda=fit_lambda,
            fix_beta=fix_beta, beta_bounds=beta_bounds,
            H0_init=H0_init, beta_init=beta_init, lam_init=lam_init,
            R0=R0, T=T,
            orient_mode=orient_mode, orient_floor=orient_floor, orient_pow=orient_pow,
            geom_mode=geom_mode
        )
    else:
        # parse beta_bounds string for per-step fit
        bb_tuple = tuple(float(x) for x in str(beta_bounds).split(","))
        et_fit_step_params_hop_redox(
            obj, csv_path, fit_lambda_per_step=fit_lambda,
            beta_bounds=bb_tuple,
            R0=R0, T=T,
            orient_mode=orient_mode, orient_floor=orient_floor, orient_pow=orient_pow,
            geom_mode=geom_mode
        )

# Tab 10: Marcus Disorder ET (MD) - Updated
def _build_tab_marcus_md(nb, add_field, add_dropdown, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Dynamic ET (MD)")

    topfile = add_csv_field(tab, "Topology (.prmtop)", 0)
    trajfile = add_csv_field(tab, "Trajectory (.nc)", 1)
    dch = add_field(tab, "Donor Chain", 2, tk.StringVar, "A")
    drs = add_field(tab, "Donor Resi", 3, tk.StringVar, "382")
    ach = add_field(tab, "Acceptor Chain", 4, tk.StringVar, "A")
    ars = add_field(tab, "Acceptor Resi", 5, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 6, ["default", "selec_atoms"], "default")
    
    # New: Theory selection
    theory_var = add_dropdown(tab, "Theory", 7, list(_engine_map.keys()), "marcus")
    
    # Common params (shifted)
    lambda_eV = add_field(tab, "λ", 8, tk.DoubleVar, DEF_LAMBDA)
    dG = add_field(tab, "ΔG", 9, tk.DoubleVar, DEF_DELTAG)
    T = add_field(tab, "T [K]", 10, tk.DoubleVar, DEF_TK)
    H0v = add_field(tab, "H0", 11, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β", 12, tk.DoubleVar, DEF_BETA)
    R0v = add_field(tab, "R0", 13, tk.DoubleVar, DEF_R0)
    
    # Theory-specific collapsible frames
    mlj_frame, mlj_toggle, _ = create_collapsible_frame(tab, "MLJ-Specific Params", 14)
    S_var = add_field(mlj_frame, "S (CSV)", 0, tk.StringVar, "1.0")
    hw_var = add_field(mlj_frame, "hw (CSV)", 1, tk.StringVar, "0.18")
    vmax_var = add_field(mlj_frame, "vmax (CSV)", 2, tk.StringVar, "10")
    mlj_frame.toggle()

    red_frame, red_toggle, _ = create_collapsible_frame(tab, "Redfield-Specific Params", 15)
    gamma_var = add_field(red_frame, "γ_deph (s^-1)", 0, tk.DoubleVar, 5e13)
    red_frame.toggle()

    tsh_frame, tsh_toggle, _ = create_collapsible_frame(tab, "TSH-Specific Params", 16)
    mass_var = add_field(tsh_frame, "mass_amu", 0, tk.DoubleVar, 100.0)
    omega_var = add_field(tsh_frame, "omega_cm", 1, tk.DoubleVar, 1000.0)
    ntraj_var = add_field(tsh_frame, "n_trajectories", 2, tk.IntVar, 100)
    tsh_frame.toggle()
    
    om, of, op = add_orient_params(tab, 17)
    plot_hist = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Show histogram", variable=plot_hist).grid(row=20, column=1, sticky="w")
    use_kubo = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Apply Kubo–Anderson correction", variable=use_kubo).grid(row=21, column=1, sticky="w")

    # Callback for theory fields
    def update_theory_fields(*args):
        th = theory_var.get().lower()
        mlj_frame.grid_remove()
        red_frame.grid_remove()
        tsh_frame.grid_remove()
        if th == 'mlj':
            mlj_frame.grid()
        elif th == 'redfield':
            red_frame.grid()
        elif th == 'tsh':
            tsh_frame.grid()

    theory_var.trace_add("write", update_theory_fields)
    update_theory_fields()

    def run_md_rate():
        # Build engine_params
        params_tk = {
            'lambda_eV': lambda_eV, 'deltaG_eV': dG, 'T_K': T,
            'H0': H0v, 'beta': betav, 'R0': R0v,
            'S_list': S_var,
            'hw_list': hw_var,'vmax_list': vmax_var,
            'gamma_deph_s1': gamma_var,
            'mass_amu': mass_var, 'omega_cm': omega_var, 'n_trajectories': ntraj_var }
        
        # Call extended et_rate_marcus (assumes updated to handle theory/params)
        et_rate_marcus_MD(
                    topfile.get(), trajfile.get(), dch.get(), drs.get(),
                    ach.get(), ars.get(),
                    orient_mode=om.get(), orient_floor=of.get(), orient_pow=op.get(),
                    geom_mode=geom_mode.get(),
                    plot_hist=plot_hist.get(), use_kubo=use_kubo.get(),
                    theory=theory_var.get(),
                    engine_params=build_engine_params_generic(params_tk, theory_var.get()) )

    tk.Button(tab, text="Run", command=run_md_rate).grid(row=22, column=0, columnspan=2, pady=5)


# Tab 11: Dynamic Bridge ET (MD) - Updated
def _build_tab_dynamic_bridge_md(nb, add_field, add_dropdown, add_dropdown_widget, add_orient_params):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Dynamic Bridge ET (MD)")

    topfile = add_csv_field(tab, "Topology (.prmtop)", 0)
    trajfile = add_csv_field(tab, "Trajectory (.nc)", 1)
    dch = add_field(tab, "Donor Chain", 2, tk.StringVar, "A")
    drs = add_field(tab, "Donor Resi", 3, tk.StringVar, "306")
    ach = add_field(tab, "Acceptor Chain", 4, tk.StringVar, "A")
    ars = add_field(tab, "Acceptor Resi", 5, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 6, ["default", "selec_atoms"], "default")
    calc_mode, calc_widget = add_dropdown_widget(tab, "Bridge Calc. Mode", 7, ["const", "pot", "gf", "hop"], "const")
    
    # New: Theory selection
    theory_var = add_dropdown(tab, "Theory", 8, list(_engine_map.keys()), "marcus")
    
    bridge_list_var = add_field(tab, "Bridges (Chain:Resi+...)", 9, tk.StringVar, "")
    emode, emode_widget = add_dropdown_widget(tab, "Energy Submode (gf/hop)", 10, ["pot", "const"], "pot")
    emode_args_var = add_field(tab, "Energy Mode Args (CSV)", 11, tk.StringVar, "")
    lambda_eV = add_field(tab, "λ", 12, tk.DoubleVar, DEF_LAMBDA)
    dG = add_field(tab, "ΔG", 13, tk.DoubleVar, DEF_DELTAG)
    T = add_field(tab, "T [K]", 14, tk.DoubleVar, DEF_TK)
    H0v = add_field(tab, "H0", 15, tk.DoubleVar, DEF_H0)
    betav = add_field(tab, "β", 16, tk.DoubleVar, DEF_BETA)
    R0v = add_field(tab, "R0", 17, tk.DoubleVar, DEF_R0)
    
    # Theory-specific frames
    mlj_frame, mlj_toggle, _ = create_collapsible_frame(tab, "MLJ-Specific Params", 18)
    S_var = add_field(mlj_frame, "S (CSV)", 0, tk.StringVar, "1.0")
    hw_var = add_field(mlj_frame, "hw (CSV)", 1, tk.StringVar, "0.18")
    vmax_var = add_field(mlj_frame, "vmax (CSV)", 2, tk.StringVar, "10")
    mlj_frame.toggle()

    red_frame, red_toggle, _ = create_collapsible_frame(tab, "Redfield-Specific Params", 19)
    gamma_var = add_field(red_frame, "γ_deph (s^-1)", 0, tk.DoubleVar, 5e13)
    red_frame.toggle()

    tsh_frame, tsh_toggle, _ = create_collapsible_frame(tab, "TSH-Specific Params", 20)
    mass_var = add_field(tsh_frame, "mass_amu", 0, tk.DoubleVar, 100.0)
    omega_var = add_field(tsh_frame, "omega_cm", 1, tk.DoubleVar, 1000.0)
    ntraj_var = add_field(tsh_frame, "n_trajectories", 2, tk.IntVar, 100)
    tsh_frame.toggle()
    
    om, of, op = add_orient_params(tab, 21)
    plot_hist = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Show histogram", variable=plot_hist).grid(row=24, column=1, sticky="w")
    seg_stats = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Segment coupling stats", variable=seg_stats).grid(row=25, column=1, sticky="w")
    seg_plot = tk.BooleanVar(value=False)
    tk.Checkbutton(tab, text="Plot segment correlation heatmap", variable=seg_plot).grid(row=26, column=1, sticky="w")
    link_bridge_and_emode(calc_mode, emode, emode_widget)

    # Callback for theory
    def update_theory_fields(*args):
        th = theory_var.get().lower()
        mlj_frame.grid_remove()
        red_frame.grid_remove()
        tsh_frame.grid_remove()
        if th == 'mlj':
            mlj_frame.grid()
        elif th == 'redfield':
            red_frame.grid()
        elif th == 'tsh':
            tsh_frame.grid()
    theory_var.trace_add("write", update_theory_fields)
    update_theory_fields()

    def run_bridge_md_rate():
        try:
            bridges_parsed = _parse_bridge_list(bridge_list_var.get())
        except Exception as e:
            print(f"Bridge parse error: {e}")
            return
        calc_mode_val = calc_mode.get().lower()
        emode_val = emode.get().lower()
        raw_args = emode_args_var.get().strip()
        emode_args_parsed = _parse_float_list(raw_args)

        # Process emode_args as per original
        if calc_mode_val in ("gf", "hop"):
            if emode_val == "pot":
                if len(emode_args_parsed) < 2 + len(bridges_parsed):
                    print("Need Eox_D, Ered_A, bridge potentials")
                    return
                EoxD, EredA = emode_args_parsed[0], emode_args_parsed[1]
                Ereds_br = emode_args_parsed[2:2 + len(bridges_parsed)]
                emode_args_final = [EoxD, EredA, Ereds_br]
            elif emode_val == "const" and calc_mode_val == "gf":
                Etun = emode_args_parsed[0]
                deltaBs = emode_args_parsed[1:] if len(emode_args_parsed) > 1 else [DEF_DeltaB] * len(bridges_parsed)
                emode_args_final = [Etun] + deltaBs
            elif emode_val == "dg" and calc_mode_val == "hop":
                emode_args_final = emode_args_parsed
            else:
                print(f"Unsupported Emode '{emode_val}' for calc '{calc_mode_val}'")
                return
        elif calc_mode_val == "const":
            emode_args_final = [float(emode_args_parsed[0])] if emode_args_parsed else [DEF_DeltaB]
        elif calc_mode_val == "pot":
            if len(emode_args_parsed) < 2 + len(bridges_parsed):
                print("Need Eox_D, Ered_A, bridge potentials")
                return
            EoxD, EredA = emode_args_parsed[0], emode_args_parsed[1]
            Ereds_br = emode_args_parsed[2:2 + len(bridges_parsed)]
            emode_args_final = [EoxD, EredA, Ereds_br]
        else:
            print(f"Unknown calc mode '{calc_mode_val}'")
            return

        # Build engine_params
        params_tk = {
            'lambda_eV': lambda_eV, 'deltaG_eV': dG, 'T_K': T,
            'H0': H0v, 'beta': betav, 'R0': R0v,
            'S_list': S_var,
            'hw_list': hw_var,'vmax_list': vmax_var,
            'gamma_deph_s1': gamma_var,
            'mass_amu': mass_var, 'omega_cm': omega_var, 'n_trajectories': ntraj_var }
        et_bridge_chain_dynamic_MD(
            topfile.get(), trajfile.get(), dch.get(), drs.get(),
            bridges_parsed, ach.get(), ars.get(), calc_mode_val, emode_val, emode_args_final,
            orient_mode=om.get(), orient_floor=of.get(), orient_pow=op.get(),
            geom_mode=geom_mode.get(),
            plot_hist=plot_hist.get(),
            do_segment_analysis=seg_stats.get(),
            segment_plot=seg_plot.get(),
            theory=theory_var.get(),
            engine_params=build_engine_params_generic(params_tk, theory_var.get()) )

    tk.Button(tab, text="Run", command=run_bridge_md_rate).grid(row=27, column=0, columnspan=2, pady=5)

#  Tab 12: Aromatic Ring Metrics
def _build_tab_aromatic_metrics(nb, add_field, add_dropdown):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Aromatic Ring Metrics")

    obj = add_field(tab, "PDB Object", 0, tk.StringVar, "1DNP")
    dchain = add_field(tab, "Donor Chain", 1, tk.StringVar, "A")
    drsi = add_field(tab, "Donor Resi", 2, tk.StringVar, "382")
    achain = add_field(tab, "Acceptor Chain", 3, tk.StringVar, "A")
    arsi = add_field(tab, "Acceptor Resi", 4, tk.StringVar, "472")
    geom_mode = add_dropdown(tab, "Geometry Mode", 5, ["default", "selec_atoms"], "default")

    def run_metrics():
        et_aromatic_metrics(
            obj.get(), dchain.get(), drsi.get(),
            achain.get(), arsi.get(), geom_mode.get()
        )

    tk.Button(tab, text="Compute Metrics & Visualize", command=run_metrics).grid(row=6, column=0, columnspan=2, pady=10)

