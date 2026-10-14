#!/usr/bin/env python
"""chequeo_final.py -- checks before submitting the final runs (login node, ~5-10 min).

    NANOGRAV_ROOT=$HOME/vec2 CONGELAR_SECULARES=PBDOT python chequeo_final.py

  1. mass files: 407 / 500 / 1476 / 200 masses, all distinct and parseable
  2. per coupling and mode: prior range fixed in rangos_prior.py, 0.005 dex bins,
     start point inside the prior, PRIOR_ACOPLE = uniforme (cotas) / loguniforme (bf)
  3. J1713+0747 residual order (chi2 per TOA ~219, not ~1863)
  4. the -3/2 x d_eta term is in the ELL1 template of each coupling
  5. DMJUMPs filled from the .par, ENTERPRISE likelihood finite, the model used
     by the chains is the ENTERPRISE one (PulsarULDM_RN), for an ELL1 and an
     eccentric pulsar
Ends in TODO OK or lists what failed.
"""
import os
import sys
import json
import subprocess

AQUI = os.path.dirname(os.path.abspath(__file__))
N_MASAS = {"masas_alpha_cotas.txt": 407, "masas_beta_cotas.txt": 500,
           "masas_vec_cotas.txt": 1476, "masas_bf.txt": 200}


def hijo(acople, modo):
    """Runs inside a subprocess (the pipelines read the environment at import)."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["PRIOR_ACOPLE"] = "uniforme" if modo == "cotas" else "loguniforme"
    sys.path.insert(0, AQUI)
    import numpy as np
    try:
        from loguru import logger as _lg; _lg.disable("pint")
    except Exception:
        pass
    import rangos_prior
    rangos_prior.exportar(acople)
    P = __import__({"alpha": "pipeline_alpha", "beta": "pipeline_beta", "vec": "pipeline_vec"}[acople])
    E = getattr(P, {"alpha": "ALPHA_EDGES", "beta": "BETA_EDGES", "vec": "G_EDGES"}[acople])
    ini = getattr(P, {"alpha": "LOG10_ALPHA_INIT", "beta": "LOG10_BETA_INIT", "vec": "LOG10_G_INIT"}[acople])
    paso = (E[-1] - E[0]) / (len(E) - 1)
    r = {"rango": [float(E[0]), float(E[-1])], "n_bins": len(E) - 1, "dex_bin": float(paso),
         "prior": P.PRIOR_ACOPLE, "version": P.VERSION_PIPELINE, "likelihood": P.LIKELIHOOD,
         "ini_ok": bool(E[0] + 0.5 < ini - 2.0 and ini + 2.0 < E[-1] - 0.5)}
    if modo == "bf":                      # the rest does not depend on the prior
        print("JSON" + json.dumps(r)); return
    # J1713 residual order
    par, tim, _c, _p = P.resolve_paths("J1713+0747")
    psr, _tm = P.load_pulsar(par, tim, convert_low_ecc=True)
    r["chi2_J1713"] = float(np.mean((np.asarray(psr.residuals) / np.asarray(psr.toaerrs)) ** 2))
    # eta term present, DMJUMPs filled, ENTERPRISE likelihood finite
    src = open(os.path.join(AQUI, {"alpha": "pipeline_alpha.py", "beta": "pipeline_beta.py",
                                   "vec": "pipeline_vec.py"}[acople]), encoding="utf-8").read()
    r["eta"] = src.count('(partials["EPS1"] - 1.5 * A1_s) * dEPS1') == 1
    rng = np.random.default_rng(7)
    r["casos"] = []
    for name, m in (("J1909-3744", 3e-20), ("J2234+0611", 1e-21)):
        d = P.preload_one(name, m)
        me = P.crear_modelo(d)
        dmj = {k: v for k, v in d["noise_norn"].items() if k.endswith("_dmjump")} if "noise_norn" in d else {}
        lls = [me.lnlikelihood(me.x0(rng)) for _ in range(5)]
        r["casos"].append([name, len(dmj), int(sum(abs(v) > 0 for v in dmj.values())),
                           bool(np.all(np.isfinite(lls))), type(me).__name__])
    print("JSON" + json.dumps(r))


def main():
    malos = []
    print("1) archivos de masas")
    for f, n in N_MASAS.items():
        ls = [l.strip() for l in open(os.path.join(AQUI, f)) if l.strip()]
        v = [float(x) for x in ls]
        ok = len(ls) == n and len(set(ls)) == n and min(v) >= 0.99e-23 and max(v) <= 1.01e-18
        print(f"   {f:24s} {len(ls):5d} masas  [{min(v):.3e} .. {max(v):.3e}]  {'OK' if ok else 'FALLA'}")
        if not ok:
            malos.append(f)
    for acople in ("alpha", "beta", "vec"):
        for modo in ("cotas", "bf"):
            print(f"2-5) {acople} {modo} ...", flush=True)
            p = subprocess.run([sys.executable, __file__, "--hijo", acople, modo],
                               capture_output=True, text=True, cwd=AQUI)
            js = [l[4:] for l in p.stdout.splitlines() if l.startswith("JSON")]
            if p.returncode != 0 or not js:
                print("   FALLA (el proceso termino con error):\n" + "\n".join(p.stderr.splitlines()[-15:]))
                malos.append(f"{acople}-{modo}"); continue
            r = json.loads(js[0])
            prior_ok = r["prior"] == ("uniforme" if modo == "cotas" else "loguniforme")
            bins_ok = abs(r["dex_bin"] - 0.005) < 1e-9
            print(f"   rango log10 [{r['rango'][0]:.3f}, {r['rango'][1]:.3f}]  {r['n_bins']} bines de "
                  f"{r['dex_bin']:.6f} dex  {'OK' if bins_ok else 'FALLA'}")
            print(f"   prior {r['prior']}  {'OK' if prior_ok else 'FALLA'};  arranque dentro del prior "
                  f"{'OK' if r['ini_ok'] else 'FALLA'};  version {r['version']}  likelihood {r['likelihood']}")
            ok = prior_ok and bins_ok and r["ini_ok"]
            if "chi2_J1713" in r:
                o = r["chi2_J1713"] < 1000
                print(f"   J1713 chi2/TOA {r['chi2_J1713']:.1f}  {'OK' if o else 'PERMUTADOS'}")
                ok &= o
                o = r["eta"]
                print(f"   termino -3/2 x deta en el template ELL1: {'OK' if o else 'FALTA'}")
                ok &= o
                for name, nd, nnz, fin, clase in r["casos"]:
                    o = nd > 0 and nnz > 0 and fin and clase == "PulsarULDM_RN"
                    print(f"   {name}: {nd} DMJUMP fijados ({nnz} no nulos), lnL ENTERPRISE finita {fin}, "
                          f"modelo {clase}  {'OK' if o else 'FALLA'}")
                    ok &= o
            if not ok:
                malos.append(f"{acople}-{modo}")
    print("\nTODO OK" if not malos else f"\nFALLA: {malos}")
    sys.exit(0 if not malos else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo":
        hijo(sys.argv[2], sys.argv[3])
    else:
        main()
