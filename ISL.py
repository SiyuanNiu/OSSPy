import os
import sys
import subprocess
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


phreeqc_exe = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\Release\phreeqc.exe"
PHREEQC_DB = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\llnl(reedited)_2021_SEP_15_Tc_Th_U_Np_Am_Se_B-Li-EPRIV-PO4.dat"

RE_INPUT = "re_step.pqi"
ADV_INPUT = "advect_step_1.pqi"
RE_STDOUT = "re_out.txt"
ADV_STDOUT = "adv_out.txt"

N_CYCLES = 10000
PRINT_EVERY = 50

COMBINED_LOG = "combined_operator_splitting_verify.log"
COMBINED_CSV = "all_selected_out_verify.csv"


def run_once(exe, in_file, std_file, db_file):
    cmd = [exe, in_file, std_file, db_file]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace"
    )
    return ret.returncode, ret.stdout, ret.stderr


def append_log(msg):
    with open(COMBINED_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def read_sel_file(path):
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#')
    except Exception:
        return pd.DataFrame()
    return df

def get_dynamic_rate(total_time):
    """
    根据总模拟时间 total_time (秒) 自动决定速率。
    """
    if total_time <= 864000:                     # 0–10天
        return 3.5e-8
    elif total_time <= 10.5 * 86400:             # 10–10.5天
        return 2.5e-8
    elif total_time <= 11 * 86400:               # 10.5–11天
        return 1e-8
    elif total_time <= 11.5 * 86400:               # 10.5–11天
        return 3e-9
    else:             # 11–11.5天
        return 5e-11

def main():
    # 检查输入文件
    for f in (RE_INPUT, ADV_INPUT):
        if not os.path.exists(f):
            print(f"[ERROR] 找不到 {f}")
            sys.exit(1)

    if os.path.exists(COMBINED_CSV):
        os.remove(COMBINED_CSV)
    combined_df = pd.DataFrame()

    with open(COMBINED_LOG, "w", encoding="utf-8") as f:
        f.write(f"# Operator-splitting (no electric field) started at {datetime.now()}\n")

    total_time_so_far = 0.0
    last_re_mtime = None
    last_adv_mtime = None

    for k in range(1, N_CYCLES + 1):
        adv_df = read_sel_file("adv_selected_out.sel")
        if not adv_df.empty and "TOTAL_TIME_GLOBAL" in adv_df.columns:
            total_time_so_far = adv_df["TOTAL_TIME_GLOBAL"].iloc[-1] * 86400.0  # 天→秒
        else:
            # 如果第一次或读取失败，默认使用上一次值
            if total_time_so_far == 0:
                total_time_so_far = 0
                append_log(f"[WARN] Step {k}: adv_selected_out.sel 读取失败，使用上一步时间 0 s")

        # === (2) 动态确定速率 ===
        current_rate = get_dynamic_rate(total_time_so_far)
        # --- EM step ---
        if k == 1:
            re_input_file = RE_INPUT
        else:
            with open(RE_INPUT, "r", encoding="utf-8") as f:
                re_lines = f.readlines()
            db_index = next(i for i, line in enumerate(re_lines)
                            if line.strip().lower().startswith("database"))

            # 2) 抽取需要保留的模块（TRANSPORT / USER_PUNCH / SELECTED_OUTPUT / DUMP）
            keep_blocks = ( "solution 0", "equilibrium_phases 0", "phases",
                            "transport", "copy", "user_punch", "selected_output", "dump")
            skip_block = "solution 1-10", "equilibrium_phases 1-10", "kinetics", "rates"

            transport_lines = []
            capture = False
            skip = False
            for line in re_lines:
                key = line.strip().lower()

                # 判断是否进入一个需要保留的块
                if key.startswith(keep_blocks):
                    capture = True
                    skip = False
                # 判断是否进入一个需要跳过的块（kinetics）
                elif key.startswith(skip_block):
                    capture = False
                    skip = True

                # 收集保留块内容
                if capture and not skip:
                    if key.startswith("-shifts") and len(line.split()) == 1:
                        transport_lines.append("    -shifts 1\n")
                    else:
                        transport_lines.append(line)

                # 退出当前块
                if key.startswith("end"):
                    if capture:
                        transport_lines.append(line)
                        capture = False
                    skip = False  # 确保下一个块正常识别

            # 重新构建文件
            new_lines = re_lines[:db_index + 1]
            new_lines.append("INCLUDE$ adv_solutions_1.pqi\n")
            new_lines.append("INCLUDE$ kinetics.pqi\n")

            # --- 自动插入新的 RATES 块 ---
            new_lines.append("\nRATES\n")
            new_lines.append("    UO2_chem\n")
            new_lines.append("-start\n")
            new_lines.append(f"10 rate = {current_rate:e}\n")
            new_lines.append("20 SAVE rate * TIME\n")
            new_lines.append("-end\n\n")
            new_lines.append("    UO2_diff\n")
            new_lines.append("-start\n")
            new_lines.append("10 A = parm(1)\n")
            new_lines.append("20 k1 = parm(2)\n")
            new_lines.append("30 k2 = parm(3)\n")
            new_lines.append("40 logK = parm(4)\n")
            new_lines.append("80 K_eq = 10^logK\n")
            new_lines.append("90 a_O2 = act(\"O2\")\n")
            new_lines.append("100 a_HCO3 = act(\"HCO3-\")\n")
            new_lines.append("110 a_UO2CO3 = act(\"UO2(CO3)3-4\")\n")
            new_lines.append("120 a_H = act(\"H+\")\n")
            new_lines.append("130 Q = (a_UO2CO3 * a_H) / (a_O2^0.5 * a_HCO3^3)\n")
            new_lines.append("140 term_thermo = (1 - Q/K_eq)\n")
            new_lines.append("150 rate =A * (k1 * a_H^0.37 * a_O2^0.31 + k2 * a_HCO3^0.35) * term_thermo\n")
            new_lines.append("160 SAVE rate * TIME\n")
            new_lines.append("-end\n\n")
            new_lines.append("    Quartz\n")
            new_lines.append("-start\n")
            new_lines.append("10 rate = 0\n")
            new_lines.append("20 SAVE rate * TIME\n")
            new_lines.append("-end\n\n")

            new_lines.extend(transport_lines)

            # 去除重复 END
            while new_lines and new_lines[-1].strip().upper() == "END":
                new_lines = new_lines[:-1]

            # === 🔍 删除两个 DUMP 块之间的多余 END ===
            dump_indices = [i for i, l in enumerate(new_lines) if l.strip().lower().startswith("dump")]
            if len(dump_indices) >= 2:
                first_dump = dump_indices[0]
                second_dump = dump_indices[1]
                # 保留第一个 END，删除两个 DUMP 之间多余的 END（如果有多个）
                cleaned = []
                end_found = False
                for i, line in enumerate(new_lines):
                    if first_dump < i < second_dump and line.strip().upper() == "END":
                        if not end_found:
                            cleaned.append(line)  # 保留第一个 END
                            end_found = True
                        else:
                            continue  # 删除多余 END
                    else:
                        cleaned.append(line)
                new_lines = cleaned

            # === 末尾确保有 END 结束整个文件 ===
            if not new_lines[-1].strip().upper() == "END":
                new_lines.append("END\n")

            re_input_file = "re_step_tmp.pqi"
            with open(re_input_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)



        rc, out, err = run_once(phreeqc_exe, re_input_file, RE_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[RE FAIL] {k}, rc={rc}\n{err}")
            sys.exit(rc)
        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] RE step OK")
        append_log(f"[{k}/{N_CYCLES}] RE step OK")

        # ========== 检查 Reaction 输出 ==========
        # === 检查 em_solutions 是否更新 ===
        if os.path.exists("re_solutions.pqi"):
            mtime = os.path.getmtime("re_solutions.pqi")
            if last_re_mtime is not None and mtime == last_re_mtime:
                raise RuntimeError(f"[Check FAIL] RE step {k} 没有更新 re_solutions.pqi (mtime 未变)")
            print(f"[Check PASS] RE step {k} 更新 re_solutions.pqi (mtime={time.ctime(mtime)})")
            last_re_mtime = mtime

        # --- ADV step ---
        rc, out, err = run_once(phreeqc_exe, ADV_INPUT, ADV_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[ADV FAIL] {k}, rc={rc}\n{err}")
            sys.exit(rc)
        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] ADV step OK")
        append_log(f"[{k}/{N_CYCLES}] ADV step OK")

        # === 检查 adv_solutions 是否更新 ===
        if os.path.exists("adv_solutions_1.pqi"):
            mtime = os.path.getmtime("adv_solutions_1.pqi")
            if last_adv_mtime is not None and mtime == last_adv_mtime:
                raise RuntimeError(f"[Check FAIL] ADV step {k} 没有更新 adv_solutions_1.pqi (mtime 未变)")
            print(f"[Check PASS] ADV step {k} 更新 adv_solutions_1.pqi (mtime={time.ctime(mtime)})")
            last_adv_mtime = mtime

        # --- 结果累计 ---
        adv_df = read_sel_file("adv_selected_out.sel")
        if not adv_df.empty:
            delta_t = adv_df["Time"].iloc[-1]
            if "TOTAL_TIME_GLOBAL" not in adv_df.columns:
                adv_df["TOTAL_TIME_GLOBAL"] = adv_df["Time"] + total_time_so_far / 86400.0
            #adv_df["TOTAL_TIME_GLOBAL"] = adv_df["Time"] + total_time_so_far
            last_row = adv_df.tail(1)
            combined_df = pd.concat([combined_df, last_row], ignore_index=True)
            total_time_so_far = adv_df["TOTAL_TIME_GLOBAL"].iloc[-1] * 86400.0  # 统一为秒单位
            #total_time_so_far += adv_df["Time"].iloc[-1]
            append_log(f"[{k:04d}] Δt={delta_t:.1f}s, Total={total_time_so_far:.1f}s "
                       f"({total_time_so_far / 86400:.3f} d)")

        combined_df.to_csv(COMBINED_CSV, index=False)

    # 绘图
    if not combined_df.empty: plt.figure(figsize=(7, 4))
    plt.plot(combined_df["TOTAL_TIME_GLOBAL"], combined_df["U_mgL"], marker='o', linestyle='-')
    plt.xlabel("Time (day)")
    plt.ylabel("U concentration (mg/L)")
    plt.title("reaction-advection case: U at outlet vs time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
    if not combined_df.empty and "TOTAL_TIME_GLOBAL" in combined_df.columns and "U_mgL" in combined_df.columns:
        df_plot = combined_df[["TOTAL_TIME_GLOBAL", "U_mgL"]].copy()
        df_plot["TOTAL_TIME_GLOBAL"] = pd.to_numeric(df_plot["TOTAL_TIME_GLOBAL"], errors="coerce")
        df_plot["U_mgL"] = pd.to_numeric(df_plot["U_mgL"], errors="coerce")
        df_plot = df_plot.dropna().sort_values("TOTAL_TIME_GLOBAL")

        # --- 步骤1：平台区间抽稀（取中点） ---
        # 若相邻点U浓度变化极小(<1e-3)，则只保留代表点
        tol = 1e-3
        diffs = np.abs(np.diff(df_plot["U_mgL"].values))
        mask = np.ones(len(df_plot), dtype=bool)
        mask[1:-1] = ~((diffs[:-1] < tol) & (diffs[1:] < tol))  # 中间点剔除
        df_plot = df_plot[mask]

        # --- 步骤2：重新采样到更稀疏的时间网格（例如200点） ---
        t = df_plot["TOTAL_TIME_GLOBAL"].values
        u = df_plot["U_mgL"].values
        t_new = np.linspace(t.min(), t.max(), 200)
        u_interp = np.interp(t_new, t, u)

        # --- 步骤3：应用 Savitzky–Golay 平滑滤波 ---
        # 窗口大小设为点数的1/10（奇数）
        window = max(5, int(len(u_interp) / 10) | 1)
        u_smooth = savgol_filter(u_interp, window_length=window, polyorder=3)

        # --- 步骤4：单调化（Isotonic Regression） ---
        # 假设物理上为“上升—下降”，自动检测趋势后单调化下降部分
        # 1) 找到峰值点
        peak_idx = np.argmax(u_smooth)
        # 2) 对峰值之后部分施加单调递减约束
        ir = IsotonicRegression(increasing=False)
        u_tail_smooth = ir.fit_transform(t_new[peak_idx:], u_smooth[peak_idx:])
        # 3) 拼接：保留上升段 + 单调下降段
        u_monotonic = np.concatenate([u_smooth[:peak_idx], u_tail_smooth])

        # 1) 在原始（抽稀后）的 u 中找出接近最大值的连续区间
        plateau_tol = 1e-10  # 用于判定数值相等的极小阈值
        u_max = np.nanmax(u)
        # 将接近 u_max 的点视为位于平顶（可用较大 tol 例如 1e-2 以放宽判定）
        plateau_mask_raw = np.isclose(u, u_max, atol=1e-10) | (u >= u_max - 1e-10)
        if plateau_mask_raw.any():
            plateau_times = t[plateau_mask_raw]
            plateau_start = plateau_times.min()
            plateau_end = plateau_times.max()
            plateau_value = u_max

            # 将 t_new 中落在 plateau 区间的值设为 plateau_value
            plateau_indices = np.where((t_new >= plateau_start) & (t_new <= plateau_end))[0]
            if plateau_indices.size > 0:
                u_monotonic[plateau_indices] = plateau_value

            try:
                from scipy.ndimage import gaussian_filter1d
                # 避免平滑改变 plateau 区域：只对非 plateau 区间分别平滑
                if plateau_mask_raw.any() and plateau_indices.size > 0:
                    left_idx = 0
                    right_idx = len(t_new) - 1
                    p0 = plateau_indices[0]
                    p1 = plateau_indices[-1]
                    # 平滑左段、右段
                    left_sm = gaussian_filter1d(u_monotonic[:p0], sigma=1.0) if p0 > 3 else u_monotonic[:p0]
                    right_sm = gaussian_filter1d(u_monotonic[p1 + 1:], sigma=1.0) if (len(u_monotonic) - (
                            p1 + 1)) > 3 else u_monotonic[p1 + 1:]
                    u_final = np.concatenate([left_sm, u_monotonic[p0:p1 + 1], right_sm])
                else:
                    u_final = gaussian_filter1d(u_monotonic, sigma=1.0)
            except Exception:
                u_final = u_monotonic  # 若无法导入 gaussian，退回原结果

        # --- 绘图 ---
        exp_data = pd.DataFrame({
        "Time": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        "U_exp": [0, 18.9, 18.7, 18.1, 17.4, 19.3, 6.25, 2.19, 1.74, 2.1, 2.18, 1.3, 0.83, 0.84, 0.79, 0.85, 0.94, 0.849, 0.882, 0.596]
        })

        # 插值模拟结果到实验时间点
        u_sim_interp = np.interp(exp_data["Time"], t_new, u_final)

        # === 步骤7：计算 R², MAE, RMSE ===
        r2 = r2_score(exp_data["U_exp"], u_sim_interp)
        mae = mean_absolute_error(exp_data["U_exp"], u_sim_interp)
        rmse = np.sqrt(mean_squared_error(exp_data["U_exp"], u_sim_interp))

        print(f"Model fitting metrics:")
        print(f"  R²   = {r2:.4f}")
        print(f"  MAE  = {mae:.3f} mg/L")
        print(f"  RMSE = {rmse:.3f} mg/L")

        plt.figure(figsize=(8, 5))
        plt.plot(t_new, u_final, '-', lw=2.5, color='tab:blue', label='Simulated (smoothed)')
        plt.scatter(exp_data["Time"], exp_data["U_exp"], s=50, color='red', label='Experimental data', zorder=3)
        plt.xlabel("Time (day)")
        plt.ylabel("U concentration (mg/L)")
        plt.title(f"U concentration at core outlet vs global time\n(R²={r2:.3f}, RMSE={rmse:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.show()

    else:
        print("No valid data for plotting.")
    '''


    print("全部循环完成。")
    append_log(f"# Finished at {datetime.now()}")




if __name__ == "__main__":
    main()
