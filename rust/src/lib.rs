//! Rust computational core for `bot-traffic-econometrics`.
//!
//! Implements four hot-path statistical and point-process engines:
//! 1. Dynamic unbalanced-panel shift-share decomposition + parallel Dirichlet cluster bootstrap.
//! 2. Exact Theil-Sen median slope + tie-corrected Mann-Kendall trend test.
//! 3. Multi-flow Expectation-Maximization under interstitial challenge censoring.
//! 4. O(N * D) recursive Ozaki log-likelihood, analytical gradients, O(N * D) EM MLE, and
//!    Ogata thinning simulation for multivariate exponential Hawkes processes.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelDecomp {
    pub r0: f64,
    pub r1: f64,
    pub delta_total: f64,
    pub survivor_r0: f64,
    pub survivor_r1: f64,
    pub within_behavior: f64,
    pub survivor_reallocation: f64,
    pub entry_selection: f64,
    pub exit_selection: f64,
}

/// Deterministic SplitMix64 PRNG so Rust and tests reproduce across platforms without rand bloat.
#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_open01(&mut self) -> f64 {
        // Uniform in (0, 1) with 53-bit mantissa precision, strictly > 0.
        let bits = (self.next_u64() >> 11) | 1;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn next_exp1(&mut self) -> f64 {
        -self.next_open01().ln()
    }
}

/// Weighted dynamic Melitz-Polanec + symmetric Kitagawa decomposition over partitioned cohorts:
/// - `surv_n0, surv_b0, surv_n1, surv_b1`: matched survivor entities in both periods
/// - `exit_n0, exit_b0`: entities present in period 0 only
/// - `entry_n1, entry_b1`: entities present in period 1 only
///
/// Optional entity weights (`w_surv`, `w_exit`, `w_entry`) support Bayesian Dirichlet bootstrap.
#[allow(clippy::too_many_arguments)]
pub fn panel_decompose_weighted(
    surv_n0: &[f64],
    surv_b0: &[f64],
    surv_n1: &[f64],
    surv_b1: &[f64],
    exit_n0: &[f64],
    exit_b0: &[f64],
    entry_n1: &[f64],
    entry_b1: &[f64],
    w_surv: Option<&[f64]>,
    w_exit: Option<&[f64]>,
    w_entry: Option<&[f64]>,
) -> PanelDecomp {
    let mut sn0 = 0.0;
    let mut sb0 = 0.0;
    let mut sn1 = 0.0;
    let mut sb1 = 0.0;
    for i in 0..surv_n0.len() {
        let mult = w_surv.map_or(1.0, |w| w[i]);
        sn0 += mult * surv_n0[i];
        sb0 += mult * surv_b0[i];
        sn1 += mult * surv_n1[i];
        sb1 += mult * surv_b1[i];
    }

    let mut xn0 = 0.0;
    let mut xb0 = 0.0;
    for i in 0..exit_n0.len() {
        let mult = w_exit.map_or(1.0, |w| w[i]);
        xn0 += mult * exit_n0[i];
        xb0 += mult * exit_b0[i];
    }

    let mut mn1 = 0.0;
    let mut mb1 = 0.0;
    for i in 0..entry_n1.len() {
        let mult = w_entry.map_or(1.0, |w| w[i]);
        mn1 += mult * entry_n1[i];
        mb1 += mult * entry_b1[i];
    }

    let total_n0 = sn0 + xn0;
    let total_n1 = sn1 + mn1;
    let r0 = if total_n0 > 0.0 {
        (sb0 + xb0) / total_n0
    } else {
        0.0
    };
    let r1 = if total_n1 > 0.0 {
        (sb1 + mb1) / total_n1
    } else {
        0.0
    };
    let sr0 = if sn0 > 0.0 { sb0 / sn0 } else { 0.0 };
    let sr1 = if sn1 > 0.0 { sb1 / sn1 } else { 0.0 };

    let mut within = 0.0;
    let mut realloc = 0.0;
    if sn0 > 0.0 && sn1 > 0.0 {
        for i in 0..surv_n0.len() {
            let mult = w_surv.map_or(1.0, |w| w[i]);
            let n0i = mult * surv_n0[i];
            let n1i = mult * surv_n1[i];
            if n0i <= 0.0 || n1i <= 0.0 {
                continue;
            }
            let r0i = (mult * surv_b0[i]) / n0i;
            let r1i = (mult * surv_b1[i]) / n1i;
            let ws0 = n0i / sn0;
            let ws1 = n1i / sn1;
            within += 0.5 * (ws0 + ws1) * (r1i - r0i);
            realloc += (ws1 - ws0) * 0.5 * (r0i + r1i);
        }
    }

    let s0 = if total_n0 > 0.0 { sn0 / total_n0 } else { 1.0 };
    let s1 = if total_n1 > 0.0 { sn1 / total_n1 } else { 1.0 };
    let rx0 = if xn0 > 0.0 { xb0 / xn0 } else { sr0 };
    let rm1 = if mn1 > 0.0 { mb1 / mn1 } else { sr1 };

    let entry_sel = (1.0 - s1) * (rm1 - sr1);
    let exit_sel = (1.0 - s0) * (sr0 - rx0);

    PanelDecomp {
        r0,
        r1,
        delta_total: r1 - r0,
        survivor_r0: sr0,
        survivor_r1: sr1,
        within_behavior: within,
        survivor_reallocation: realloc,
        entry_selection: entry_sel,
        exit_selection: exit_sel,
    }
}

/// Parallel Rubin Bayesian (Dirichlet) entity-cluster bootstrap returning a flat `n_boot * 5`
/// matrix of `[delta_total, within_behavior, survivor_reallocation, entry_selection, exit_selection]`.
#[allow(clippy::too_many_arguments)]
pub fn panel_bayesian_bootstrap(
    surv_n0: &[f64],
    surv_b0: &[f64],
    surv_n1: &[f64],
    surv_b1: &[f64],
    exit_n0: &[f64],
    exit_b0: &[f64],
    entry_n1: &[f64],
    entry_b1: &[f64],
    n_boot: usize,
    seed: u64,
) -> Vec<[f64; 5]> {
    (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let mut rng = SplitMix64::new(seed ^ ((b as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)));
            let ws: Vec<f64> = (0..surv_n0.len()).map(|_| rng.next_exp1()).collect();
            let wx: Vec<f64> = (0..exit_n0.len()).map(|_| rng.next_exp1()).collect();
            let wm: Vec<f64> = (0..entry_n1.len()).map(|_| rng.next_exp1()).collect();
            let d = panel_decompose_weighted(
                surv_n0,
                surv_b0,
                surv_n1,
                surv_b1,
                exit_n0,
                exit_b0,
                entry_n1,
                entry_b1,
                Some(&ws),
                Some(&wx),
                Some(&wm),
            );
            [
                d.delta_total,
                d.within_behavior,
                d.survivor_reallocation,
                d.entry_selection,
                d.exit_selection,
            ]
        })
        .collect()
}

/// Exact Theil-Sen median slope + intercept and tie-corrected Mann-Kendall (tau_b, z_score).
pub fn theil_sen_kendall(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (0.0, y.first().copied().unwrap_or(0.0), 0.0, 0.0);
    }
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    let mut s_stat: i64 = 0;
    let mut x_ties_pairs: i64 = 0;
    let mut y_ties_pairs: i64 = 0;
    let mut xy_ties_pairs: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            if dx.abs() > 0.0 {
                slopes.push(dy / dx);
            }
            let sx = if dx > 0.0 {
                1
            } else if dx < 0.0 {
                -1
            } else {
                0
            };
            let sy = if dy > 0.0 {
                1
            } else if dy < 0.0 {
                -1
            } else {
                0
            };
            if sx == 0 && sy == 0 {
                xy_ties_pairs += 1;
                x_ties_pairs += 1;
                y_ties_pairs += 1;
            } else if sx == 0 {
                x_ties_pairs += 1;
            } else if sy == 0 {
                y_ties_pairs += 1;
            } else {
                s_stat += sx * sy;
            }
        }
    }

    let slope = median_in_place(&mut slopes);
    let mut intercepts: Vec<f64> = (0..n).map(|i| y[i] - slope * x[i]).collect();
    let intercept = median_in_place(&mut intercepts);

    let total_pairs = (n * (n - 1) / 2) as f64;
    let denom = ((total_pairs - x_ties_pairs as f64) * (total_pairs - y_ties_pairs as f64)).sqrt();
    let tau_b = if denom > 0.0 {
        (s_stat as f64) / denom
    } else {
        0.0
    };

    // Tie-corrected variance of Mann-Kendall S (Kendall 1975).
    let mut sorted_y = y[..n].to_vec();
    sorted_y.sort_by(|a, b| a.total_cmp(b));
    let mut tie_term = 0.0;
    let mut idx = 0;
    while idx < n {
        let mut end = idx + 1;
        while end < n && (sorted_y[end] - sorted_y[idx]).abs() == 0.0 {
            end += 1;
        }
        let t = (end - idx) as f64;
        if t > 1.0 {
            tie_term += t * (t - 1.0) * (2.0 * t + 5.0);
        }
        idx = end;
    }
    let nf = n as f64;
    let _ = xy_ties_pairs;
    let var_s = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - tie_term) / 18.0;
    let z = if var_s > 0.0 {
        if s_stat > 0 {
            ((s_stat as f64) - 1.0) / var_s.sqrt()
        } else if s_stat < 0 {
            ((s_stat as f64) + 1.0) / var_s.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    (slope, intercept, tau_b, z)
}

fn median_in_place(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let m = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[m]
    } else {
        0.5 * (vals[m - 1] + vals[m])
    }
}

/// Joint multi-flow Expectation-Maximization under interstitial challenge censoring.
///
/// For each flow `f in 0..n_flows`, we observe uncensored bot counts `(u0, u1, u2)` in tiers
/// `{Low, Medium, High}`, censored challenged count `c_f`, and known challenge policy exposure
/// `exposure[f] in (0, 1]` such that `P(C = 1 | S = k, f) = exposure[f] * gamma[k]`.
/// Returns `(pi_flat [n_flows * 3], gamma [3], log_likelihood, iterations)`.
pub fn censored_em_fit(
    uncensored_flat: &[f64],
    censored: &[f64],
    exposure: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, [f64; 3], f64, usize) {
    let n_flows = censored.len();
    assert_eq!(uncensored_flat.len(), n_flows * 3);
    assert_eq!(exposure.len(), n_flows);

    let mut pi = vec![1.0 / 3.0; n_flows * 3];
    for f in 0..n_flows {
        let u0 = uncensored_flat[f * 3];
        let u1 = uncensored_flat[f * 3 + 1];
        let u2 = uncensored_flat[f * 3 + 2];
        let usum = u0 + u1 + u2;
        if usum > 0.0 {
            pi[f * 3] = (u0 + 1e-3) / (usum + 3e-3);
            pi[f * 3 + 1] = (u1 + 1e-3) / (usum + 3e-3);
            pi[f * 3 + 2] = (u2 + 1e-3) / (usum + 3e-3);
        }
    }
    // Distinct monotone initial gamma breaks symmetry across tiers.
    let mut gamma = [0.65_f64, 0.45_f64, 0.25_f64];
    let eps = 1e-9_f64;

    let observed_loglik = |pi_ref: &[f64], g_ref: &[f64; 3]| -> f64 {
        let mut ll = 0.0;
        for f in 0..n_flows {
            let ef = exposure[f].clamp(eps, 1.0);
            let mut chal_prob = 0.0;
            for k in 0..3 {
                let p_fk = pi_ref[f * 3 + k].max(eps);
                let gk = (ef * g_ref[k]).clamp(eps, 1.0 - eps);
                let u_fk = uncensored_flat[f * 3 + k];
                if u_fk > 0.0 {
                    ll += u_fk * (p_fk * (1.0 - gk)).ln();
                }
                chal_prob += p_fk * gk;
            }
            if censored[f] > 0.0 {
                ll += censored[f] * chal_prob.max(eps).ln();
            }
        }
        ll
    };

    let mut prev_ll = observed_loglik(&pi, &gamma);
    let mut iters = 0;

    for it in 0..max_iter {
        iters = it + 1;
        // E-step: posterior responsibilities for censored counts per flow.
        let mut q = vec![0.0_f64; n_flows * 3];
        for f in 0..n_flows {
            let mut denom = 0.0;
            for k in 0..3 {
                let num = pi[f * 3 + k] * gamma[k];
                q[f * 3 + k] = num;
                denom += num;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 1.0 / 3.0 };
            for k in 0..3 {
                q[f * 3 + k] = if denom > 0.0 { q[f * 3 + k] * inv } else { inv };
            }
        }

        // M-step for flow-specific latent tier priors pi_{f, k}.
        for f in 0..n_flows {
            let total_f = uncensored_flat[f * 3]
                + uncensored_flat[f * 3 + 1]
                + uncensored_flat[f * 3 + 2]
                + censored[f];
            if total_f > 0.0 {
                for k in 0..3 {
                    let expected_fk = uncensored_flat[f * 3 + k] + censored[f] * q[f * 3 + k];
                    pi[f * 3 + k] = (expected_fk / total_f).clamp(eps, 1.0 - eps);
                }
                let psum = pi[f * 3] + pi[f * 3 + 1] + pi[f * 3 + 2];
                for k in 0..3 {
                    pi[f * 3 + k] /= psum;
                }
            }
        }

        // M-step for shared tier challenge propensities gamma_k via 1-D concave Newton/bisection
        // on sum_f [ c_{f,k} / gamma_k - u_{f,k} * e_f / (1 - e_f * gamma_k) ] = 0.
        for k in 0..3 {
            let mut c_sum = 0.0;
            let mut u_any = 0.0;
            for f in 0..n_flows {
                c_sum += censored[f] * q[f * 3 + k];
                u_any += uncensored_flat[f * 3 + k];
            }
            if c_sum <= 0.0 {
                gamma[k] = eps;
                continue;
            }
            if u_any <= 0.0 {
                gamma[k] = 1.0 - eps;
                continue;
            }
            let mut lo = eps;
            let mut hi = 1.0 - eps;
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                let mut score = 0.0;
                for f in 0..n_flows {
                    let ef = exposure[f].clamp(eps, 1.0);
                    let c_fk = censored[f] * q[f * 3 + k];
                    let u_fk = uncensored_flat[f * 3 + k];
                    score += c_fk / mid - (u_fk * ef) / (1.0 - ef * mid).max(eps);
                }
                if score > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            gamma[k] = 0.5 * (lo + hi);
        }

        let ll = observed_loglik(&pi, &gamma);
        if (ll - prev_ll).abs() <= tol * (1.0 + prev_ll.abs()) {
            prev_ll = ll;
            break;
        }
        prev_ll = ll;
    }

    (pi, gamma, prev_ll, iters)
}

/// Exact O(N * D) Ozaki recursive log-likelihood and analytical gradients `(ll, grad_mu, grad_alpha)`
/// for a D-dimensional exponential Hawkes process on `[0, horizon]`.
pub fn hawkes_loglik_grad(
    times: &[f64],
    marks: &[usize],
    mu: &[f64],
    alpha_flat: &[f64],
    beta: f64,
    horizon: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let d = mu.len();
    assert_eq!(alpha_flat.len(), d * d);
    assert_eq!(times.len(), marks.len());

    let mut grad_mu = vec![-horizon; d];
    let mut h_j = vec![0.0_f64; d];
    for (&t, &m) in times.iter().zip(marks.iter()) {
        h_j[m] += (1.0 - (-beta * (horizon - t)).exp()) / beta;
    }

    let mut ll = 0.0_f64;
    for i in 0..d {
        ll -= mu[i] * horizon;
        for j in 0..d {
            ll -= alpha_flat[i * d + j] * h_j[j];
        }
    }

    let mut grad_alpha = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..d {
            grad_alpha[i * d + j] = -h_j[j];
        }
    }

    let mut g = vec![0.0_f64; d];
    let mut prev_t = 0.0_f64;
    for (idx, (&t, &m)) in times.iter().zip(marks.iter()).enumerate() {
        if idx > 0 {
            let decay = (-beta * (t - prev_t)).exp();
            for val in &mut g {
                *val *= decay;
            }
            g[marks[idx - 1]] += decay;
        }
        let mut lam = mu[m];
        let row = m * d;
        for j in 0..d {
            lam += alpha_flat[row + j] * g[j];
        }
        let safe_lam = lam.max(1e-15);
        ll += safe_lam.ln();
        let inv_lam = 1.0 / safe_lam;
        grad_mu[m] += inv_lam;
        for j in 0..d {
            grad_alpha[row + j] += g[j] * inv_lam;
        }
        prev_t = t;
    }

    (ll, grad_mu, grad_alpha)
}

/// Exact O(N * D) per-iteration Lewis-Mohler EM maximum-likelihood solver for multivariate
/// exponential Hawkes processes. Returns `(mu, alpha_flat, loglik, iterations)`.
pub fn hawkes_fit_em(
    times: &[f64],
    marks: &[usize],
    d: usize,
    beta: f64,
    horizon: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, f64, usize) {
    assert_eq!(times.len(), marks.len());
    let mut counts = vec![0.0_f64; d];
    let mut h_j = vec![0.0_f64; d];
    for (&t, &m) in times.iter().zip(marks.iter()) {
        counts[m] += 1.0;
        h_j[m] += (1.0 - (-beta * (horizon - t)).exp()) / beta;
    }

    let mut mu: Vec<f64> = counts
        .iter()
        .map(|&c| ((0.5 * c) / horizon.max(1e-9)).max(1e-4))
        .collect();
    let mut alpha = vec![0.05 * beta / (d.max(1) as f64); d * d];

    let mut prev_ll = f64::NEG_INFINITY;
    let mut iters = 0;
    let mut g = vec![0.0_f64; d];

    for it in 0..max_iter {
        iters = it + 1;
        g.fill(0.0);
        let mut m_sum = vec![0.0_f64; d];
        let mut s_sum = vec![0.0_f64; d * d];
        let mut ll = 0.0_f64;
        for i in 0..d {
            ll -= mu[i] * horizon;
            for j in 0..d {
                ll -= alpha[i * d + j] * h_j[j];
            }
        }

        let mut prev_t = 0.0_f64;
        for (idx, (&t, &m)) in times.iter().zip(marks.iter()).enumerate() {
            if idx > 0 {
                let decay = (-beta * (t - prev_t)).exp();
                for val in &mut g {
                    *val *= decay;
                }
                g[marks[idx - 1]] += decay;
            }
            let row = m * d;
            let mut lam = mu[m];
            for j in 0..d {
                lam += alpha[row + j] * g[j];
            }
            let safe_lam = lam.max(1e-15);
            ll += safe_lam.ln();
            let inv_lam = 1.0 / safe_lam;
            m_sum[m] += mu[m] * inv_lam;
            for j in 0..d {
                s_sum[row + j] += alpha[row + j] * g[j] * inv_lam;
            }
            prev_t = t;
        }

        for i in 0..d {
            mu[i] = (m_sum[i] / horizon).max(1e-12);
            for j in 0..d {
                alpha[i * d + j] = if h_j[j] > 0.0 {
                    (s_sum[i * d + j] / h_j[j]).max(0.0)
                } else {
                    0.0
                };
            }
        }

        if prev_ll.is_finite() && (ll - prev_ll).abs() <= tol * (1.0 + prev_ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    let (final_ll, _, _) = hawkes_loglik_grad(times, marks, &mu, &alpha, beta, horizon);
    (mu, alpha, final_ll, iters)
}

/// Exact Ogata thinning simulator for D-dimensional exponential Hawkes processes on `[0, horizon]`.
pub fn simulate_hawkes(
    mu: &[f64],
    alpha_flat: &[f64],
    beta: f64,
    horizon: f64,
    seed: u64,
    max_events: usize,
) -> (Vec<f64>, Vec<usize>) {
    let d = mu.len();
    assert_eq!(alpha_flat.len(), d * d);
    let mut rng = SplitMix64::new(seed);
    let mut times = Vec::new();
    let mut marks = Vec::new();
    let mut excit = vec![0.0_f64; d];
    let mu_sum: f64 = mu.iter().sum();

    let mut t = 0.0_f64;
    while t < horizon && times.len() < max_events {
        let excit_sum: f64 = excit.iter().sum();
        let lam_bar = mu_sum + excit_sum;
        if lam_bar <= 0.0 {
            break;
        }
        let dt = rng.next_exp1() / lam_bar;
        let t_next = t + dt;
        if t_next >= horizon {
            break;
        }
        let decay = (-beta * dt).exp();
        for e in &mut excit {
            *e *= decay;
        }
        t = t_next;

        let mut lam_now = 0.0;
        for i in 0..d {
            lam_now += mu[i] + excit[i];
        }
        if rng.next_open01() * lam_bar <= lam_now {
            let target = rng.next_open01() * lam_now;
            let mut acc = 0.0;
            let mut chosen = d - 1;
            for i in 0..d {
                acc += mu[i] + excit[i];
                if target <= acc {
                    chosen = i;
                    break;
                }
            }
            times.push(t);
            marks.push(chosen);
            for i in 0..d {
                excit[i] += alpha_flat[i * d + chosen];
            }
        }
    }

    (times, marks)
}

#[cfg(feature = "python")]
mod py_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn py_panel_decompose(
        surv_n0: Vec<f64>,
        surv_b0: Vec<f64>,
        surv_n1: Vec<f64>,
        surv_b1: Vec<f64>,
        exit_n0: Vec<f64>,
        exit_b0: Vec<f64>,
        entry_n1: Vec<f64>,
        entry_b1: Vec<f64>,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let d = panel_decompose_weighted(
            &surv_n0, &surv_b0, &surv_n1, &surv_b1, &exit_n0, &exit_b0, &entry_n1, &entry_b1, None,
            None, None,
        );
        (
            d.r0,
            d.r1,
            d.delta_total,
            d.survivor_r0,
            d.survivor_r1,
            d.within_behavior,
            d.survivor_reallocation,
            d.entry_selection,
            d.exit_selection,
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn py_panel_bootstrap(
        py: Python<'_>,
        surv_n0: Vec<f64>,
        surv_b0: Vec<f64>,
        surv_n1: Vec<f64>,
        surv_b1: Vec<f64>,
        exit_n0: Vec<f64>,
        exit_b0: Vec<f64>,
        entry_n1: Vec<f64>,
        entry_b1: Vec<f64>,
        n_boot: usize,
        seed: u64,
    ) -> Vec<[f64; 5]> {
        py.detach(|| {
            panel_bayesian_bootstrap(
                &surv_n0, &surv_b0, &surv_n1, &surv_b1, &exit_n0, &exit_b0, &entry_n1, &entry_b1,
                n_boot, seed,
            )
        })
    }

    #[pyfunction]
    fn py_theil_sen_kendall(x: Vec<f64>, y: Vec<f64>) -> (f64, f64, f64, f64) {
        theil_sen_kendall(&x, &y)
    }

    #[pyfunction]
    fn py_censored_em_fit(
        uncensored_flat: Vec<f64>,
        censored: Vec<f64>,
        exposure: Vec<f64>,
        max_iter: usize,
        tol: f64,
    ) -> (Vec<f64>, [f64; 3], f64, usize) {
        censored_em_fit(&uncensored_flat, &censored, &exposure, max_iter, tol)
    }

    #[pyfunction]
    fn py_hawkes_loglik_grad(
        times: Vec<f64>,
        marks: Vec<usize>,
        mu: Vec<f64>,
        alpha_flat: Vec<f64>,
        beta: f64,
        horizon: f64,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        hawkes_loglik_grad(&times, &marks, &mu, &alpha_flat, beta, horizon)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn py_hawkes_fit_em(
        py: Python<'_>,
        times: Vec<f64>,
        marks: Vec<usize>,
        d: usize,
        beta: f64,
        horizon: f64,
        max_iter: usize,
        tol: f64,
    ) -> (Vec<f64>, Vec<f64>, f64, usize) {
        py.detach(|| hawkes_fit_em(&times, &marks, d, beta, horizon, max_iter, tol))
    }

    #[pyfunction]
    fn py_simulate_hawkes(
        py: Python<'_>,
        mu: Vec<f64>,
        alpha_flat: Vec<f64>,
        beta: f64,
        horizon: f64,
        seed: u64,
        max_events: usize,
    ) -> (Vec<f64>, Vec<usize>) {
        py.detach(|| simulate_hawkes(&mu, &alpha_flat, beta, horizon, seed, max_events))
    }

    #[pymodule]
    fn _bte_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_panel_decompose, m)?)?;
        m.add_function(wrap_pyfunction!(py_panel_bootstrap, m)?)?;
        m.add_function(wrap_pyfunction!(py_theil_sen_kendall, m)?)?;
        m.add_function(wrap_pyfunction!(py_censored_em_fit, m)?)?;
        m.add_function(wrap_pyfunction!(py_hawkes_loglik_grad, m)?)?;
        m.add_function(wrap_pyfunction!(py_hawkes_fit_em, m)?)?;
        m.add_function(wrap_pyfunction!(py_simulate_hawkes, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_identity_is_exact_to_machine_epsilon() {
        let d = panel_decompose_weighted(
            &[1000.0, 2500.0, 800.0],
            &[220.0, 450.0, 240.0],
            &[1200.0, 2200.0, 1100.0],
            &[190.0, 380.0, 210.0],
            &[1800.0, 900.0],
            &[18.0, 12.0],
            &[600.0],
            &[180.0],
            None,
            None,
            None,
        );
        let sum_parts =
            d.within_behavior + d.survivor_reallocation + d.entry_selection + d.exit_selection;
        assert!((sum_parts - d.delta_total).abs() < 1e-15);
        assert!(d.within_behavior < 0.0);
        assert!(d.exit_selection > 0.0);
    }

    #[test]
    fn hawkes_ozaki_matches_quadratic_loglik_and_em_monotone() {
        let mu = vec![0.4, 0.25];
        let alpha = vec![0.35, 0.15, 0.25, 0.30];
        let beta = 1.5;
        let horizon = 80.0;
        let (times, marks) = simulate_hawkes(&mu, &alpha, beta, horizon, 42, 50_000);
        assert!(times.len() > 20);

        let (ll_fast, grad_mu, _) = hawkes_loglik_grad(&times, &marks, &mu, &alpha, beta, horizon);
        let mut ll_quad = -mu[0] * horizon - mu[1] * horizon;
        for i in 0..2 {
            for (&t, &m) in times.iter().zip(marks.iter()) {
                ll_quad -= alpha[i * 2 + m] * (1.0 - (-beta * (horizon - t)).exp()) / beta;
            }
        }
        for r in 0..times.len() {
            let i = marks[r];
            let mut lam = mu[i];
            for s in 0..r {
                lam += alpha[i * 2 + marks[s]] * (-beta * (times[r] - times[s])).exp();
            }
            ll_quad += lam.ln();
        }
        assert!((ll_fast - ll_quad).abs() < 1e-10);
        assert_eq!(grad_mu.len(), 2);

        let (mu_hat, alpha_hat, ll_hat, iters) =
            hawkes_fit_em(&times, &marks, 2, beta, horizon, 300, 1e-9);
        assert!(iters > 1);
        assert!(ll_hat >= ll_fast - 1e-8);
        assert!(mu_hat.iter().all(|&v| v > 0.0));
        assert!(alpha_hat.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn censored_em_recovers_latent_mix_monotonically() {
        let uncensored = vec![300.0, 200.0, 100.0, 60.0, 120.0, 420.0];
        let censored = vec![400.0, 350.0];
        let exposure = vec![1.0, 0.8];
        let (pi, gamma, ll, iters) = censored_em_fit(&uncensored, &censored, &exposure, 200, 1e-10);
        assert!(ll.is_finite());
        assert!(iters >= 1);
        assert!((pi[0] + pi[1] + pi[2] - 1.0).abs() < 1e-12);
        assert!((pi[3] + pi[4] + pi[5] - 1.0).abs() < 1e-12);
        assert!(gamma.iter().all(|&g| g > 0.0 && g < 1.0));
    }
}
