"""
plant_calibrated.py
FT plant model driven by CHGNet adsorption energies – no promoter multiplier table.
Activity uses a volcano relative to the reference catalyst's E_CO.
α depends on (E_CO - E_H) relative to the reference.
Calibrated to the Shanghai SARI pilot at a single point.
"""
import numpy as np
import pandas as pd

# ── Physical constants ──────────────────────────────────────────────────────
VOLCANO_GAMMA = 0.15          # eV^{-2}, width of relative activity volcano

def _mw_paraffin(n):
    return 12.011 * n + 1.008 * (2 * n + 2)

def asf_distribution(alpha, n_max=30):
    ns = np.arange(1, n_max + 1)
    xn = (1 - alpha) * alpha ** (ns - 1)
    xn /= xn.sum()
    return ns, xn

def mass_selectivity(alpha):
    ns, xn = asf_distribution(alpha)
    total_mass = sum(xn[i] * _mw_paraffin(ns[i]) for i in range(len(ns)))
    saf_mass   = sum(xn[i] * _mw_paraffin(ns[i]) for i in range(len(ns)) if 8 <= ns[i] <= 16)
    return saf_mass / total_mass

class CO2_to_SAF_Plant:
    def __init__(
        self,
        T_rwgs: float = 673,
        P_bar: float = 20.0,
        T_ft: float = 603.0,
        catalyst_mass_g: float = 1.0,
        recycle_ratio: float = 0.6,
        rwgs_co2_conversion: float = 0.70,
        rwgs_water_removal_fraction: float = 0.9,
        recycle_species: tuple = ("CO", "H2", "CH4"),
        # Kinetic parameters (to be calibrated)
        ref_co_consumption: float = 0.068,
        h2_co_ratio: float = 2.1,
        ch4_selectivity: float = 0.05,
        coke_carbon_fraction: float = 0.0002,
        # Reference catalyst descriptors (used for relative scaling)
        ref_E_CO: float = -1.42,
        ref_E_H: float = -0.28,
        # Current catalyst descriptors
        E_CO: float = -1.42,
        E_H: float = -0.28,
        E_O: float = -0.84,
        E_OH: float = -0.45,
    ):
        self.T_rwgs = T_rwgs
        self.P_bar = P_bar
        self.T_ft = T_ft
        self.catalyst_mass_g = catalyst_mass_g
        self.recycle_ratio = recycle_ratio
        self.rwgs_co2_conversion = rwgs_co2_conversion
        self.rwgs_water_removal_fraction = rwgs_water_removal_fraction
        self.recycle_species = recycle_species
        self.ref_co_consumption = ref_co_consumption
        self.h2_co_ratio = h2_co_ratio
        self.ch4_selectivity = ch4_selectivity
        self.coke_carbon_fraction = coke_carbon_fraction

        self.ref_E_CO = ref_E_CO
        self.ref_E_H = ref_E_H
        self.E_CO = E_CO
        self.E_H = E_H
        self.E_O = E_O
        self.E_OH = E_OH

        self.coke = 0.0
        self.activity = 1.0
        self.time_on_stream = 0.0

    # ── Relative activity factor (volcano centred at reference E_CO) ───────
    def _activity_factor(self):
        """Activity = exp(-γ * (E_CO - ref_E_CO)²)"""
        delta = self.E_CO - self.ref_E_CO
        return np.exp(-VOLCANO_GAMMA * delta * delta)

    # ── Descriptor‑based α (relative to reference (E_CO - E_H)) ────────────
    def _alpha(self):
        """α = α_ref + slope * [(E_CO-E_H) - (ref_E_CO - ref_E_H)]"""
        ref_delta = self.ref_E_CO - self.ref_E_H
        curr_delta = self.E_CO - self.E_H
        # Literature-based slope: more negative delta (stronger CO vs H) increases α
        # Here we assume α_ref = 0.78 for the reference catalyst at 603 K
        alpha = 0.78 + 0.04 * (curr_delta - ref_delta)
        # Temperature effect (universal for FT: α decreases with T)
        alpha -= 0.0005 * (self.T_ft - 603)
        # Pressure effect (higher pressure → higher α)
        alpha += 0.01 * np.log(self.P_bar / 20.0)
        return np.clip(alpha, 0.60, 0.85)

    # ── Reactor steps (unchanged from your version, correct) ───────────────
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

        # Base CO consumption from reference rate * activity factor * deactivation
        co_consumed = self.ref_co_consumption * self._activity_factor() * self.activity

        # Add temperature factor (Arrhenius)
        T_ref = 603.0
        T_factor = np.exp(-80000 / 8.314 * (1/self.T_ft - 1/T_ref))
        # Add pressure factor (sqrt dependence)
        P_ref = 20.0
        P_factor = np.sqrt(self.P_bar / P_ref)

        co_consumed = co_consumed * T_factor * P_factor

        # ← CAP: never consume more than 80% of inlet CO (kinetic limit)
        co_consumed = min(co_consumed, co_in * 0.80)

        # Rest of the method unchanged...
        gas_out = dict(feed)
        gas_out["CO"] = co_in - co_consumed
        gas_out["H2"] = max(0.0, feed.get("H2", 0.0) - co_consumed * self.h2_co_ratio)
        gas_out["H2O"] = feed.get("H2O", 0.0) + co_consumed
        gas_out["CH4"] = feed.get("CH4", 0.0) + co_consumed * self.ch4_selectivity

        alpha = self._alpha()
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

        return gas_out, coke_formed_g, {
            "saf_mass_kg_h": saf_mass_kg_h,
            "selectivity": selectivity,
            "alpha": alpha
        }

    def deactivate(self, coke_formed_g, dt_hours):
        self.coke += coke_formed_g
        self.time_on_stream += dt_hours
        self.activity = max(0.2, np.exp(-1e-4 * self.time_on_stream))

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
            # Rebuild feed: fresh + recycle (purge discarded)
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


# ── Calibration: find ref_co_consumption that matches pilot at reference catalyst ──
if __name__ == "__main__":
    # Load descriptors for the reference catalyst (K‑promoted Fe(110))
    desc_df = pd.read_csv("data/processed/catalyst_descriptors.csv")
    desc_df["facet"] = desc_df["facet"].astype(str)
    ref_row = desc_df[(desc_df["facet"] == "110") & (desc_df["promoter"] == "K")]
    if ref_row.empty:
        raise RuntimeError("No descriptor found for reference catalyst Fe(110)/K")
    ref = ref_row.iloc[0]

    target_sty = 252.7
    # Fresh feed composition (CO2:H2 = 1:3) – mass flow that will be re‑calibrated later
    fresh_feed = {"CO2": 0.1191, "H2": 3 * 0.1191}

    # Binary search for ref_co_consumption
    low, high = 0.001, 0.5
    best_val, best_err = None, 1e9
    for _ in range(30):
        mid = (low + high) / 2
        plant = CO2_to_SAF_Plant(
            T_rwgs=673, P_bar=20, T_ft=603, catalyst_mass_g=1.0,
            ref_E_CO=ref["E_CO"], ref_E_H=ref["E_H"],
            E_CO=ref["E_CO"], E_H=ref["E_H"], E_O=ref["E_O"], E_OH=ref["E_OH"],
            ref_co_consumption=mid
        )
        res = plant.run_simulation(fresh_feed)
        err = res["STY_mg_gcat_h"] - target_sty
        if abs(err) < best_err:
            best_err = abs(err)
            best_val = mid
        if err > 0:
            high = mid
        else:
            low = mid

    print(f"Calibrated ref_co_consumption = {best_val:.5f}")
    plant = CO2_to_SAF_Plant(
        T_rwgs=673, P_bar=20, T_ft=603, catalyst_mass_g=1.0,
        ref_E_CO=ref["E_CO"], ref_E_H=ref["E_H"],
        E_CO=ref["E_CO"], E_H=ref["E_H"], E_O=ref["E_O"], E_OH=ref["E_OH"],
        ref_co_consumption=best_val
    )
    res = plant.run_simulation(fresh_feed)
    print(f"STY = {res['STY_mg_gcat_h']:.2f} mg/g/h  (target {target_sty})")
    print(f"SAF selectivity = {res['SAF_selectivity']:.4f}")