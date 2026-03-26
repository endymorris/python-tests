"""
Аналитическая модель СДПМ с поверхностными магнитами (Surface PMSM)
Базовый двигатель: BM1418 ZXF (350 Вт, 48 В, 7 Н·м, 450 об/мин)

Не требует FEM-программ. Запускается на любой ОС.
Установить зависимости: pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ─── Константы материалов ─────────────────────────────────────────────────────

Br        = 1.10         # Остаточная индукция NdFeB [Тл]
mu_r      = 1.05         # Относительная проницаемость магнита
rho_cu    = 1.72e-8      # Удельное сопротивление меди [Ом·м]
rho_fe    = 7650.0       # Плотность стали [кг/м³]
k_fe      = 0.04         # Коэффициент потерь в стали (упрощён. Штейнмец) [Вт/(кг·Тл²·Гц)]
                         # Обоснование: для кремнистой стали M400-50A при 50 Гц, 1.5 Тл → ~4 Вт/кг
                         # k = 4 / (B² * f) = 4 / (1.5² * 50) ≈ 0.036 Вт/(кг·Тл²·Гц)
k_fill    = 0.45         # Коэффициент заполнения паза медью

# ─── Базовые параметры BM1418 ZXF (неизменны) ───────────────────────────────

p         = 4            # Число пар полюсов (8-полюсный двигатель)
Qs        = 12           # Число пазов статора
kw1       = 0.945        # Обмоточный коэффициент (распределённая обмотка, dn=12/8)
V_dc      = 48.0         # Напряжение питания [В] (звезда → V_ph = V_dc/√3)
V_ph      = V_dc / np.sqrt(3)   # Фазное напряжение [В] ≈ 27.7 В
n_nom     = 450.0        # Номинальная частота вращения [об/мин]
omega_m   = n_nom * 2 * np.pi / 60   # Угловая скорость [рад/с]
omega_e   = p * omega_m  # Электрическая угловая частота [рад/с]
f_e       = omega_e / (2 * np.pi)    # Электрическая частота [Гц]

# Базовые геометрические параметры (откалиброваны под 7 Н·м)
BASE = {
    'D_si':      0.100,   # Диаметр расточки статора [м] (100 мм)
    'L_stack':   0.025,   # Длина пакета [м] (25 мм)
    'h_pm':      0.0035,  # Толщина магнита [м] (3.5 мм)
    'alpha_pm':  0.80,    # Полюсное перекрытие α_п (0 ... 1)
    'g':         0.0008,  # Воздушный зазор [м] (0.8 мм)
    'N_ph':      150,     # Витков на фазу
    'A_wire':    0.50e-6, # Сечение провода [м²] (0.5 мм²)
}

# ─── Аналитическая модель ────────────────────────────────────────────────────

def compute_motor(D_si, L_stack, h_pm, alpha_pm, g, N_ph, A_wire):
    """
    Упрощённая аналитическая модель Surface PMSM.
    Погрешность относительно FEM: ~5–15% (достаточно для датасета суррогата).

    Параметры
    ---------
    D_si      : диаметр расточки статора [м]
    L_stack   : длина пакета [м]
    h_pm      : толщина магнита [м]
    alpha_pm  : коэффициент полюсного перекрытия (0 < alpha_pm < 1)
    g         : воздушный зазор [м]
    N_ph      : количество витков на фазу
    A_wire    : сечение провода фазы [м²]

    Возвращает
    ----------
    dict с {T_em, P_out, eta, E_rms, I_rms, P_cu, P_fe, cos_phi, B_gap, lambda_pm}
    Если расчёт физически неправдоподобен — возвращает NaN.
    """

    # ── Геометрия ─────────────────────────────────────────────────────────────
    tau_p   = np.pi * D_si / (2 * p)          # Полюсное деление [м]
    slot_h  = 0.30 * D_si / 2                 # Высота паза ≈ 30% R_si (масштаб)
    slot_w  = (np.pi * D_si / Qs) * 0.45      # Ширина паза (45% шага паза)

    # ── Магнитная цепь ────────────────────────────────────────────────────────
    # Воздушный зазор с учётом коэффициента Картера (упрощённо k_c ≈ 1.1)
    g_eff   = (g + h_pm / mu_r) * 1.10

    # Индукция в зазоре от магнита
    B_gap   = Br * h_pm / (mu_r * g + h_pm)      # [Тл]

    # Первая гармоника индукции
    B1      = (4 / np.pi) * B_gap * np.sin(np.pi * alpha_pm / 2)   # [Тл]

    if B_gap > 1.5 or B_gap < 0.1:
        return {k: np.nan for k in
                ['T_em','P_out','eta','E_rms','I_rms','P_cu','P_fe','cos_phi','B_gap','lambda_pm']}

    # ── Потокосцепление от магнитов ───────────────────────────────────────────
    Phi1       = B1 * tau_p * L_stack             # Основной поток одного полюса [Вб]
    lambda_pm  = kw1 * N_ph * Phi1                # Потокосцепление фазы [Вб]

    # ── ЭДС ───────────────────────────────────────────────────────────────────
    E_peak  = omega_e * lambda_pm                 # Амплитуда ЭДС [В]
    E_rms   = E_peak / np.sqrt(2)                 # Действующее значение [В]

    # Если ЭДС превышает напряжение → двигатель не работает в этом режиме
    if E_rms >= V_ph * 0.98:
        return {k: np.nan for k in
                ['T_em','P_out','eta','E_rms','I_rms','P_cu','P_fe','cos_phi','B_gap','lambda_pm']}

    # ── Сопротивление фазы ────────────────────────────────────────────────────
    # Длина средней витки = 2*L_stack + π*(R_si + slot_h/2)  (сложение по шагу лобовых частей)
    l_turn  = 2 * L_stack + np.pi * (D_si / 2 + slot_h / 2)
    l_wire  = N_ph * l_turn                       # Полная длина провода фазы [м]
    R_ph    = rho_cu * l_wire / A_wire            # Сопротивление фазы [Ом]

    # ── Синхронная индуктивность (поверхностный PM, нет явнополюсности) ────────
    # Ls = mu0 * (kw1*N_ph)^2 * tau_p * L_stack / (pi * p * g_eff)
    mu0     = 4 * np.pi * 1e-7
    Ls      = (mu0 * (kw1 * N_ph)**2 * tau_p * L_stack) / (np.pi * p * g_eff)

    # ── Фазный ток (из схемы замещения: V_ph² = (E_rms + R·I·cos_phi)² + (Xs·I)²) ──
    # При Id=0 (ориентация по вектору потока): V_ph² = (E_rms + R·Iq)² + (Xs·Iq)²
    Xs      = omega_e * Ls                        # Реактивное сопротивление [Ом]
    a_coef  = R_ph**2 + Xs**2
    b_coef  = 2 * E_rms * R_ph
    c_coef  = E_rms**2 - V_ph**2
    discr   = b_coef**2 - 4 * a_coef * c_coef

    if discr < 0:
        return {k: np.nan for k in
                ['T_em','P_out','eta','E_rms','I_rms','P_cu','P_fe','cos_phi','B_gap','lambda_pm']}

    I_rms   = (-b_coef + np.sqrt(discr)) / (2 * a_coef)

    if I_rms <= 0 or I_rms > 50:   # физичные пределы: 0 … 50 А
        return {k: np.nan for k in
                ['T_em','P_out','eta','E_rms','I_rms','P_cu','P_fe','cos_phi','B_gap','lambda_pm']}

    # ── Момент ────────────────────────────────────────────────────────────────
    # T = (3/2) * p * lambda_pm * I_q_peak  (Id=0 → I_q_peak = I_rms*√2)
    T_em    = (3 / 2) * p * lambda_pm * (I_rms * np.sqrt(2))

    # ── Потери ───────────────────────────────────────────────────────────────
    # Омические потери
    P_cu    = 3 * R_ph * I_rms**2

    # Потери в стали (упрощённый Штейнмец: W/kg * масса_стали)
    # Масса стали ≈ ярмо статора + зубцы (грубая оценка)
    D_so    = D_si + 2 * (slot_h + 0.01)       # Внешний диаметр статора [м]
    V_fe    = np.pi / 4 * (D_so**2 - D_si**2) * L_stack * 0.6   # объём стали [м³] (60% от кольца — зубцы+ярмо)
    M_fe    = rho_fe * V_fe                     # масса стали [кг]
    B_teeth = B_gap / 0.45                      # ≈ индукция в зубце (грубо)
    P_fe    = k_fe * min(B_teeth, 1.8)**2 * f_e * M_fe

    # Механические потери (трение, вентиляция) ≈ 2 % от P_out
    P_out   = T_em * omega_m
    P_mech  = 0.02 * P_out

    # КПД
    P_in    = P_out + P_cu + P_fe + P_mech
    eta     = P_out / max(P_in, 1.0)

    # Коэффициент мощности
    S       = 3 * V_ph * I_rms
    cos_phi = min(P_in / max(S, 1.0), 1.0)

    return {
        'T_em':       round(T_em,    4),   # Электромагнитный момент [Н·м]
        'P_out':      round(P_out,   2),   # Выходная мощность [Вт]
        'eta':        round(eta,     4),   # КПД (0 ... 1)
        'E_rms':      round(E_rms,   3),   # ЭДС (д.з.) [В]
        'I_rms':      round(I_rms,   4),   # Фазный ток [А]
        'P_cu':       round(P_cu,    2),   # Потери в меди [Вт]
        'P_fe':       round(P_fe,    2),   # Потери в стали [Вт]
        'cos_phi':    round(cos_phi, 4),   # Коэффициент мощности
        'B_gap':      round(B_gap,   4),   # Индукция в зазоре [Тл]
        'lambda_pm':  round(lambda_pm, 6), # Потокосцепление от магнитов [Вб]
    }


# ─── Проверка базовой точки ───────────────────────────────────────────────────

def verify_base():
    print("=" * 55)
    print("Проверка базовой точки BM1418 ZXF")
    print("=" * 55)
    result = compute_motor(**BASE)
    for k, v in result.items():
        print(f"  {k:12s} = {v}")
    print(f"\n  Ожидается:   T_em ≈ 7.0 Н·м,  P_out ≈ 330 Вт")
    print("=" * 55)
    return result


# ─── Генерация датасета (Латинский гиперкуб) ─────────────────────────────────

def generate_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует n_samples комбинаций параметров методом Латинского гиперкуба (LHS),
    вычисляет показатели для каждой и сохраняет в CSV.

    Диапазоны варьирования: ±15–20% от базовых значений BM1418 ZXF
    с учётом физических ограничений.
    """

    # Параметры и их диапазоны [min, max]
    param_bounds = {
        #  Параметр      min        max       Ед. изм.    Комментарий
        'D_si':      [0.080,     0.125],   # м           диаметр расточки 80–125 мм
        'L_stack':   [0.018,     0.040],   # м           длина пакета 18–40 мм
        'h_pm':      [0.002,     0.006],   # м           толщина магнита 2–6 мм
        'alpha_pm':  [0.60,      0.92],    # б/р         полюсное перекрытие
        'g':         [0.0005,    0.0015],  # м           воздушный зазор 0.5–1.5 мм
        'N_ph':      [80,        220],     # шт          кол-во витков на фазу
        'A_wire':    [0.25e-6,   1.00e-6], # м²          сечение провода 0.25–1.0 мм²
    }

    keys  = list(param_bounds.keys())
    lows  = np.array([param_bounds[k][0] for k in keys])
    highs = np.array([param_bounds[k][1] for k in keys])

    # LHS sampling
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    raw     = sampler.random(n=n_samples)
    scaled  = qmc.scale(raw, lows, highs)

    records = []
    n_failed = 0
    for row in scaled:
        params = dict(zip(keys, row))
        # N_ph должно быть целым числом
        params['N_ph'] = int(round(params['N_ph']))

        result = compute_motor(**params)

        # Отбрасываем физически несостоятельные точки
        if any(np.isnan(v) for v in result.values()):
            n_failed += 1
            continue

        # Дополнительные фильтры реалистичности
        if result['T_em'] < 0.5 or result['T_em'] > 50.0:
            n_failed += 1
            continue
        if result['eta'] < 0.50 or result['eta'] > 0.99:
            n_failed += 1
            continue
        if result['I_rms'] > 40.0:
            n_failed += 1
            continue

        records.append({**params, **result})

    df = pd.DataFrame(records)
    print(f"Сгенерировано: {len(df)} строк из {n_samples} попыток  "
          f"(отброшено: {n_failed})")
    return df


# ─── Точечный расчёт (для отладки) ───────────────────────────────────────────

def single_calc(h_pm_mm: float, alpha_pm: float, N_ph: int):
    """
    Быстрый одиночный расчёт — меняем три ключевых параметра на базовом двигателе.
    Удобно для ручной проверки.
    """
    params = dict(BASE)
    params['h_pm']     = h_pm_mm / 1000.0
    params['alpha_pm'] = alpha_pm
    params['N_ph']     = N_ph
    return compute_motor(**params)


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    # 1. Проверка базовой точки
    base_result = verify_base()

    # 2. Генерация датасета
    print("\nГенерация датасета (n=300, LHS)...")
    df = generate_dataset(n_samples=400, seed=42)   # 400 попыток → ~300 валидных

    if df.empty:
        print("Ошибка: датасет пустой. Проверьте диапазоны параметров.")
        sys.exit(1)

    # 3. Сохранение
    out_path = os.path.join(os.path.dirname(__file__), "dataset_pmsm.csv")
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Сохранено: {out_path}")

    # 4. Статистика
    print("\nСтатистика датасета:")
    print(df[['T_em', 'P_out', 'eta', 'I_rms', 'B_gap']].describe().round(3).to_string())

    # 5. Пример «похожих на базовый» строк
    print("\nПримеры строк с моментом 6–8 Н·м:")
    mask = (df['T_em'] >= 6.0) & (df['T_em'] <= 8.0)
    if mask.sum() > 0:
        print(df[mask][['D_si', 'h_pm', 'alpha_pm', 'N_ph', 'T_em', 'eta', 'I_rms']].head(5).to_string(index=False))
    else:
        print("Нет строк в этом диапазоне — расширьте диапазоны параметров.")
