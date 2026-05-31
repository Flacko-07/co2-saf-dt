"""
plant_calibrated.py
FT plant model – variable selectivity (T‑ and promoter‑dependent).
Uses mass selectivity of C8‑C16 (correctly matches the Shanghai target).
"""
import numpy as np

# ── Helper: molecular weight of paraffin C_n H_{2n+2} ──────────────────────
def _mw_paraffin(n):
    return 12.011 * n + 1.008 * (2 * n + 2)

# ── ASF distribution (mole fractions) ──────────────────────────────────────
def asf_distribution(alpha, n_max=30):
    ns = np.arange(1, n_max + 1)
    xn = (1 - alpha) * alpha ** (ns - 1)
    xn /= xn.sum()                     # normalise (numerical safety)
    return ns, xn

# ── Mass selectivity of C8‑C16 ─────────────────────────────────────────────
def mass_selectivity(alpha):
    ns, xn = asf_distribution(alpha)
    total_mass = sum(xn[i] * _mw_paraffin(ns[i]) for i in range(len(ns)))
    saf_mass   = sum(xn[i] * _mw_paraffin(ns[i]) for i in range(len(ns)) if 8 <= ns[i] <= 16)
    return saf_mass / total_mass

# ── Calibrated α function ──────────────────────────────────────────────────
# α at reference (603 K, promoter_multiplier=1.15) is chosen so that
# mass_selectivity(α) = 0.375 → α ≈ 0.735
REF_ALPHA = 0.795               # gives mass selectivity 0.375
def alpha_from_conditions(T, promoter_multiplier):
    base = REF_ALPHA - 0.0005 * (T - 603) + 0.01 * (promoter_multiplier - 1.15)
    return np.clip(base, 0.60, 0.85)

# ── Plant model class ──────────────────────────────────────────────────────
class CO2_to_SAF_Plant:
    def __init__(self, T_rwgs=673, P_bar=20.0, T_ft=603.0, catalyst_mass_g=1.0,
                 recycle_ratio=0.6, ft_co_conversion=0.51,
                 h2_co_consumption_ratio=2.1, ch4_selectivity=0.05,
                 coke_carbon_fraction=0.0002, promoter_multiplier=1.0,
                 rwgs_water_removal_fraction=0.9,
                 recycle_species=("CO", "H2", "CH4"),
                 rwgs_co2_conversion=0.70):
        self.T_rwgs = T_rwgs
        self.P_bar = P_bar
        self.T_ft = T_ft
        self.catalyst_mass_g = catalyst_mass_g
        self.recycle_ratio = recycle_ratio
        self.ft_co_conversion = ft_co_conversion * promoter_multiplier
        self.h2_co_consumption_ratio = h2_co_consumption_ratio
        self.ch4_selectivity = ch4_selectivity
        self.coke_carbon_fraction = coke_carbon_fraction
        self.rwgs_water_removal_fraction = rwgs_water_removal_fraction
        self.recycle_species = recycle_species
        self.rwgs_co2_conversion = rwgs_co2_conversion
        self.promoter_multiplier = promoter_multiplier
        self.coke = 0.0
        self.activity = 1.0
        self.time_on_stream = 0.0

    def rwgs_reactor(self, feed):
        out = dict(feed)
        co2_in = feed.get("CO2", 0.0)
        h2_in  = feed.get("H2", 0.0)
        co_produced = co2_in * self.rwgs_co2_conversion
        h2_consumed = co_produced
        out["CO2"] = co2_in - co_produced
        out["CO"]  = out.get("CO", 0.0) + co_produced
        out["H2"]  = max(0.0, h2_in - h2_consumed)
        out["H2O"] = out.get("H2O", 0.0) + co_produced
        return out

    def remove_rwgs_water(self, stream):
        out = dict(stream)
        water_in = max(out.get("H2O", 0.0), 0.0)
        water_removed = self.rwgs_water_removal_fraction * water_in
        out["H2O"] = water_in - water_removed
        return out, water_removed

    def ft_reactor(self, feed, dt_hours=1.0):
        co_in = max(feed.get("CO", 0.0), 0.0)
        if co_in <= 1e-12:
            return dict(feed), 0.0, {"saf_mass_kg_h": 0.0, "selectivity": 0.0}

        effective_conversion = self.ft_co_conversion * self.activity
        co_consumed = co_in * effective_conversion

        gas_out = dict(feed)
        gas_out["CO"] = co_in - co_consumed
        gas_out["H2"] = max(0.0, feed.get("H2", 0.0) - co_consumed * self.h2_co_consumption_ratio)
        gas_out["H2O"] = feed.get("H2O", 0.0) + co_consumed
        gas_out["CH4"] = feed.get("CH4", 0.0) + co_consumed * self.ch4_selectivity

        # ASF product distribution (mass selectivity)
        alpha = alpha_from_conditions(self.T_ft, self.promoter_multiplier)
        ns, xn = asf_distribution(alpha)
        avg_c = float(np.dot(ns, xn))
        total_hc_mol_h = co_consumed / avg_c

        saf_mass_kg_h = 0.0
        for n, x in zip(ns, xn):
            flow_mol_h = total_hc_mol_h * x
            mass_kg_h = flow_mol_h * _mw_paraffin(n) / 1000.0
            if 8 <= n <= 16:
                saf_mass_kg_h += mass_kg_h

        selectivity = mass_selectivity(alpha)
        coke_formed_g = co_consumed * self.coke_carbon_fraction * 12.011 * dt_hours

        return gas_out, coke_formed_g, {"saf_mass_kg_h": saf_mass_kg_h,
                                         "selectivity": selectivity,
                                         "alpha": alpha}

    def deactivate(self, coke_formed_g, dt_hours):
        self.coke += coke_formed_g
        self.time_on_stream += dt_hours
        self.activity = max(0.2, np.exp(-1e-4 * self.time_on_stream)) if self.time_on_stream > 0 else 1.0

    def recycle_stream(self, outlet, recycle_ratio):
        recycle = {}
        purge = dict(outlet)
        for species in self.recycle_species:
            if species in outlet:
                amount = outlet[species] * recycle_ratio
                recycle[species] = amount
                purge[species] = outlet[species] - amount
        for sp in outlet:
            if sp not in self.recycle_species and sp not in purge:
                purge[sp] = outlet[sp]
        return recycle, purge

    def run_simulation(self, fresh_feed, simulation_hours=1.0, dt_hours=1.0):
        n_steps = max(1, int(simulation_hours / dt_hours))
        stream = dict(fresh_feed)
        total_saf_kg = 0.0
        avg_selectivity = 0.0
        for _ in range(n_steps):
            rwgs_out = self.rwgs_reactor(stream)
            stream, _ = self.remove_rwgs_water(rwgs_out)
            ft_out, coke_formed, metrics = self.ft_reactor(stream, dt_hours)
            stream = ft_out
            self.deactivate(coke_formed, dt_hours)
            recycle, purge = self.recycle_stream(stream, self.recycle_ratio)
            stream = {}
            for sp in set(list(fresh_feed.keys()) + list(recycle.keys())):
                stream[sp] = fresh_feed.get(sp, 0.0) + recycle.get(sp, 0.0)
            total_saf_kg += metrics["saf_mass_kg_h"] * dt_hours
            avg_selectivity += metrics["selectivity"]
        avg_selectivity /= n_steps
        sty_mg_g_h = (total_saf_kg / (self.catalyst_mass_g * simulation_hours)) * 1e6
        return {"STY_mg_gcat_h": sty_mg_g_h,
                "SAF_selectivity": avg_selectivity,
                "coke_total_g": self.coke}


# ── Calibration (find feed rate that hits 252.7 mg/g/h) ────────────────────
if __name__ == "__main__":
    plant = CO2_to_SAF_Plant(T_rwgs=673, P_bar=20, T_ft=603,
                             catalyst_mass_g=1.0, ft_co_conversion=0.51,
                             promoter_multiplier=1.15)   # K promoter
    target_sty = 252.7
    best_feed, best_err = None, 1e9
    for co2 in np.linspace(0.001, 1.0, 10000):
        fresh = {"CO2": co2, "H2": 3 * co2}
        res = plant.run_simulation(fresh, simulation_hours=1.0)
        err = abs(res["STY_mg_gcat_h"] - target_sty)
        if err < best_err:
            best_err = err
            best_feed = co2
            if err < 0.05:
                break
    print(f"Calibrated feed: CO₂ = {best_feed:.6f} mol/h/gcat")
    fresh = {"CO2": best_feed, "H2": 3 * best_feed}
    res = plant.run_simulation(fresh)
    print(f"STY = {res['STY_mg_gcat_h']:.2f} mg/g/h  (target {target_sty})")
    print(f"Selectivity = {res['SAF_selectivity']:.4f}  (target ~0.375)")