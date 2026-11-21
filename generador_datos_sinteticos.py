"""
Generador de datos SINTETICOS para el Proyecto Final CCI (AndesVision AI).
Reproduce el esquema de solicitudes.csv, monitoreo.csv y solicitudes_validacion.csv.
Parametros del grupo: tau=230, t1=100, t2=450, pi=0.006, c0=2.
La semilla de generacion (SEED_SINT) NO es la S del grupo: es solo para fabricar estos archivos.
"""
import numpy as np, pandas as pd, json, sys

SEED_SINT = 777
TAU, T1, T2, PI, C0 = 230, 100, 450, 0.006, 2

def simular_ventana(rng, dur=1200, lam=14.0, g_scale=1.0, p_fail=(0.08, 0.06),
                    seg_incidente=(780, 960), mult_inc=(1.7, 1.5, 2.0), sucio=True,
                    gpu_factor=1.0):
    # ---- llegadas: Poisson no homogeneo por tramos (regimen anomalo en seg_incidente)
    tiempos = []
    t0, t1_ = seg_incidente
    for a, b, m in [(0, t0, 1.0), (t0, t1_, mult_inc[0]), (t1_, dur, 1.0)]:
        n = rng.poisson(lam * m * (b - a))
        tiempos.append(np.sort(rng.uniform(a, b, n)))
    t = np.concatenate(tiempos)
    n = len(t)
    inc = (t >= t0) & (t < t1_)                       # indicador de regimen anomalo
    # ---- tamano (lognormal), cache dependiente del tamano
    tam = np.exp(rng.normal(np.log(150), 0.75, n))
    clase = np.where(tam < T1, 0, np.where(tam < T2, 1, 2))
    p_cache = np.array([0.55, 0.28, 0.08])[clase]
    cache = (rng.uniform(size=n) < p_cache).astype(int)
    version = np.where(rng.uniform(size=n) < 0.5, "A", "B")
    # ---- red (lognormal, cola pesada) y GPU (media lineal en tam, gamma heterocedastica)
    red = np.exp(rng.normal(np.log(28), 0.45, n)) * np.where(inc, 1.3, 1.0)
    media_g = (20 + 0.22 * tam) * np.where(version == "B", 0.95, 1.0) * gpu_factor
    media_g = media_g * np.where(inc, mult_inc[1], 1.0)
    shape = 7.0
    gpu = rng.gamma(shape, media_g / shape)
    gpu[cache == 1] = 0.0
    # ---- intentos (geometrica truncada en 3 fallos => abandono)
    pf = np.where(version == "A", p_fail[0], p_fail[1]) * np.where(inc, 1.8, 1.0)
    fallos = np.zeros(n, int)
    vivo = np.ones(n, bool)
    for _ in range(3):
        f = (rng.uniform(size=n) < pf) & vivo
        fallos += f
        vivo &= f
    intentos = np.minimum(fallos + 1, 3)
    exito = (fallos < 3).astype(int)
    # ---- cola FIFO con C0 GPUs (solo no cacheadas)
    libre = np.zeros(C0)
    espera = np.zeros(n)
    for i in range(n):
        if cache[i] == 0:
            k = np.argmin(libre)
            ini = max(libre[k], t[i])
            espera[i] = (ini - t[i]) * 1000
            libre[k] = ini + gpu[i] / 1000
    backoff = (intentos - 1) * rng.uniform(20, 60, n)
    lat = red + espera + gpu + backoff
    df = pd.DataFrame(dict(id_solicitud=np.arange(1, n + 1), t_llegada_s=t.round(3),
        tam_kb=tam.round(2), cache_hit=cache, version=version, red_ms=red.round(2),
        gpu_ms=gpu.round(2), latencia_ms=lat.round(2), intentos=intentos, exito=exito))
    clave = {}
    if sucio:
        d = df.copy()
        def pick(k): return rng.choice(len(d), k, replace=False)
        i = pick(25);  d.loc[i, "gpu_ms"] = -d.loc[i, "gpu_ms"].abs().clip(lower=5); clave["gpu_ms_negativo"] = d.loc[i, "id_solicitud"].tolist()
        j = pick(40);  j = [x for x in j if d.loc[x, "cache_hit"] == 1][:20]
        d.loc[j, "gpu_ms"] = rng.uniform(30, 120, len(j)).round(2); clave["cache_con_gpu"] = d.loc[j, "id_solicitud"].tolist()
        k = [x for x in pick(80) if d.loc[x, "cache_hit"] == 0][:20]
        d.loc[k, "gpu_ms"] = 0.0; clave["nocache_gpu_cero"] = d.loc[k, "id_solicitud"].tolist()
        m = pick(30);  d.loc[m, "latencia_ms"] = (d.loc[m, "red_ms"] + d.loc[m, "gpu_ms"].clip(lower=0)) * 0.5; clave["latencia_menor_que_red_mas_gpu"] = d.loc[m, "id_solicitud"].tolist()
        q = pick(10);  d.loc[q, "tam_kb"] = rng.choice([0.0, -15.0, 9.5e5, 4.0e4], 10); clave["tam_invalido_o_extremo"] = d.loc[q, "id_solicitud"].tolist()
        r_ = pick(15); d.loc[r_, "intentos"] = rng.choice([0, 4, 5], 15); clave["intentos_fuera_de_rango"] = d.loc[r_, "id_solicitud"].tolist()
        s_ = [x for x in pick(60) if d.loc[x, "intentos"] in (1, 2)][:15]
        d.loc[s_, "exito"] = 0; clave["exito_0_con_menos_de_3_intentos"] = d.loc[s_, "id_solicitud"].tolist()
        u = pick(12);  d.loc[u, "red_ms"] = np.nan; clave["red_ms_nan"] = d.loc[u, "id_solicitud"].tolist()
        v = pick(10);  d.loc[v, "version"] = d.loc[v, "version"].str.lower(); clave["version_minuscula"] = d.loc[v, "id_solicitud"].tolist()
        w = pick(8);   d.loc[w, "t_llegada_s"] = rng.choice([-3.2, 1203.5, 1500.0], 8); clave["t_llegada_fuera_de_ventana"] = d.loc[w, "id_solicitud"].tolist()
        dup = d.sample(12, random_state=int(rng.integers(1e6)))
        clave["filas_duplicadas_id"] = dup["id_solicitud"].tolist()
        d = pd.concat([d, dup]).sort_values("id_solicitud", kind="stable").reset_index(drop=True)
        df = d
    return df, clave, (t0, t1_)

def monitoreo(rng, n=6000):
    # incidentes en rachas (cadena de Markov), prevalencia ~ 12%
    inc = np.zeros(n, int); s = 0
    for i in range(n):
        s = 1 if ((s and rng.uniform() < 0.92) or (not s and rng.uniform() < 0.0136)) else 0
        inc[i] = s
    # factor latente comun (ruido de plataforma) => D1 y D2 NO condicionalmente independientes
    z = (rng.uniform(size=n) < 0.10).astype(int)
    p1 = np.where(inc == 1, 0.86, 0.04) + 0.30 * z * (inc == 0) - 0.0 * z
    p2 = np.where(inc == 1, 0.72, 0.03) + 0.28 * z * (inc == 0)
    D1 = (rng.uniform(size=n) < np.clip(p1, 0, 1)).astype(int)
    D2 = (rng.uniform(size=n) < np.clip(p2, 0, 1)).astype(int)
    return pd.DataFrame(dict(minuto=np.arange(1, n + 1), D1=D1, D2=D2, incidente=inc))

if __name__ == "__main__":
    out = "/mnt/user-data/outputs/datos_sinteticos/"
    rng = np.random.default_rng(SEED_SINT)
    sol, clave, seg = simular_ventana(rng)
    sol.to_csv(out + "solicitudes.csv", index=False)
    mon = monitoreo(rng); mon.to_csv(out + "monitoreo.csv", index=False)
    rng2 = np.random.default_rng(SEED_SINT + 1)
    val, _, _ = simular_ventana(rng2, lam=14.7, p_fail=(0.10, 0.07), seg_incidente=(1200, 1200), gpu_factor=1.06, sucio=False)
    val.to_csv(out + "solicitudes_validacion.csv", index=False)
    clave["_regimen_anomalo_segundos"] = list(seg)
    json.dump(clave, open(out + "CLAVE_respuestas_NO_ABRIR_hasta_terminar_A1.json", "w"), indent=1)
    print(len(sol), len(mon), len(val), mon.incidente.mean())
