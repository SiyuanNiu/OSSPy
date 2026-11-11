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

    if total_time <= 864000:                     
        return 3.5e-8
    elif total_time <= 10.5 * 86400:             
        return 2.5e-8
    elif total_time <= 11 * 86400:               
        return 1e-8
    elif total_time <= 11.5 * 86400:               
        return 3e-9

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
            if total_time_so_far == 0:
                total_time_so_far = 0
                append_log(f"[WARN] Step {k}: adv_selected_out.sel 读取失败，使用上一步时间 0 s")

        current_rate = get_dynamic_rate(total_time_so_far)
        # --- EM step ---
        if k == 1:
            re_input_file = RE_INPUT
        else:
            with open(RE_INPUT, "r", encoding="utf-8") as f:
                re_lines = f.readlines()
            db_index = next(i for i, line in enumerate(re_lines)
                            if line.strip().lower().startswith("database"))

            keep_blocks = ( "solution 0", "equilibrium_phases 0", "phases",
                            "transport", "copy", "user_punch", "selected_output", "dump")
            skip_block = "solution 1-10", "equilibrium_phases 1-10", "kinetics", "rates"

            transport_lines = []
            capture = False
            skip = False
            for line in re_lines:
                key = line.strip().lower()

                if key.startswith(keep_blocks):
                    capture = True
                    skip = False
                elif key.startswith(skip_block):
                    capture = False
                    skip = True

                if capture and not skip:
                    if key.startswith("-shifts") and len(line.split()) == 1:
                        transport_lines.append("    -shifts 1\n")
                    else:
                        transport_lines.append(line)

                if key.startswith("end"):
                    if capture:
                        transport_lines.append(line)
                        capture = False
                    skip = False  

            new_lines = re_lines[:db_index + 1]
            new_lines.append("INCLUDE$ adv_solutions_1.pqi\n")
            new_lines.append("INCLUDE$ kinetics.pqi\n")

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

            while new_lines and new_lines[-1].strip().upper() == "END":
                new_lines = new_lines[:-1]

            dump_indices = [i for i, l in enumerate(new_lines) if l.strip().lower().startswith("dump")]
            if len(dump_indices) >= 2:
                first_dump = dump_indices[0]
                second_dump = dump_indices[1]
                cleaned = []
                end_found = False
                for i, line in enumerate(new_lines):
                    if first_dump < i < second_dump and line.strip().upper() == "END":
                        if not end_found:
                            cleaned.append(line) 
                            end_found = True
                        else:
                            continue  
                    else:
                        cleaned.append(line)
                new_lines = cleaned

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

        if os.path.exists("adv_solutions_1.pqi"):
            mtime = os.path.getmtime("adv_solutions_1.pqi")
            if last_adv_mtime is not None and mtime == last_adv_mtime:
                raise RuntimeError(f"[Check FAIL] ADV step {k} 没有更新 adv_solutions_1.pqi (mtime 未变)")
            print(f"[Check PASS] ADV step {k} 更新 adv_solutions_1.pqi (mtime={time.ctime(mtime)})")
            last_adv_mtime = mtime

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

    # Drawing
    if not combined_df.empty: plt.figure(figsize=(7, 4))
    plt.plot(combined_df["TOTAL_TIME_GLOBAL"], combined_df["U_mgL"], marker='o', linestyle='-')
    plt.xlabel("Time (day)")
    plt.ylabel("U concentration (mg/L)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("全部循环完成。")
    append_log(f"# Finished at {datetime.now()}")




if __name__ == "__main__":
    main()
