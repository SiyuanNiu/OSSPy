import os
import sys
import subprocess
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


phreeqc_exe = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\Release\phreeqc.exe"
PHREEQC_DB = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\llnl(reedited)_2021_SEP_15_Tc_Th_U_Np_Am_Se_B-Li-EPRIV-PO4.dat"
EM_INPUT = "em_step.pqi"
ADV_INPUT = "advect_step.pqi"
EM_STDOUT = "em_out.txt"
ADV_STDOUT = "adv_out.txt"

N_CYCLES = 10000


PRINT_EVERY = 50


APPEND_COMBINED_LOG = True
COMBINED_LOG = "combined_operator_splitting.log"
COMBINED_CSV = "all_selected_out.csv"
# ===========================================


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

    if total_time <= 3 * 86400:                     
        return 6e-8
    elif total_time <= 6 * 86400:            
        return 3e-8
    elif total_time <= 8 * 86400:              
        return 8e-9
    else:             
        return 3e-9


def main():
    for f in (EM_INPUT, ADV_INPUT):
        if not os.path.exists(f):
            print(f"[ERROR] 找不到 {f}")
            sys.exit(1)

    if os.path.exists(COMBINED_CSV):
        os.remove(COMBINED_CSV)
    combined_df = pd.DataFrame()

    with open(COMBINED_LOG, "w", encoding="utf-8") as f:
        f.write(f"# Operator-splitting (no electric field) started at {datetime.now()}\n")

    total_time_so_far = 0.0
    last_em_mtime = None
    last_adv_mtime = None

    for k in range(1, N_CYCLES + 1):
        adv_df = read_sel_file("adv_selected_out_1.sel")
        if not adv_df.empty and "TOTAL_TIME_GLOBAL" in adv_df.columns:
            total_time_so_far = adv_df["TOTAL_TIME_GLOBAL"].iloc[-1] * 86400.0  
        else:
            if total_time_so_far == 0:
                total_time_so_far = 0
                append_log(f"[WARN] Step {k}: adv_selected_out.sel 

        current_rate = get_dynamic_rate(total_time_so_far)
        # --- EM step ---
        if k == 1:
            em_input_file = EM_INPUT
        else:
            with open(EM_INPUT, "r", encoding="utf-8") as f:
                em_lines = f.readlines()
            db_index = next(i for i, line in enumerate(em_lines)
                            if line.strip().lower().startswith("database"))

            keep_blocks = ("solution 0", "equilibrium_phases 0", "solution 11", "phases",
                           "transport", "copy", "user_punch", "selected_output", "dump")
            skip_block = "solution 1-10", "equilibrium_phases 1-10", "kinetics 1-10", "rates"

            transport_lines = []
            capture = False
            skip = False
            for line in em_lines:
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

            new_lines = em_lines[:db_index + 1]
            new_lines.append("INCLUDE$ adv_solutions.pqi\n")
            new_lines.append("INCLUDE$ kinetics_em.pqi\n")

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

            em_input_file = "em_step_tmp.pqi"
            with open(em_input_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        rc, out, err = run_once(phreeqc_exe, em_input_file, EM_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[EM FAIL] 循环 {k}/{N_CYCLES} 失败，returncode={rc}\nSTDERR:\n{err}")
            append_log(f"[EM FAIL] cycle={k}, rc={rc}\n{err}\n")
            sys.exit(rc)

        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] EM step OK")
        append_log(f"[{k}/{N_CYCLES}] EM step OK")

        #strip_potential("em_solutions.pqi", "em_solutions_ADV.pqi")

        # ---- ADV step ----
        rc, out, err = run_once(phreeqc_exe, ADV_INPUT, ADV_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[ADV FAIL] 循环 {k}/{N_CYCLES} 失败，returncode={rc}\nSTDERR:\n{err}")
            print("STDOUT:\n", out)
            print("STDERR:\n", err)
            append_log(f"[ADV FAIL] cycle={k}, rc={rc}\n{err}\n")
            sys.exit(rc)
        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] ADV step OK")
        append_log(f"[{k}/{N_CYCLES}] ADV step OK")

        adv_df = read_sel_file("adv_selected_out_1.sel")

        if not adv_df.empty:
            delta_t = adv_df["Time"].iloc[-1]
            if "TOTAL_TIME_GLOBAL" not in adv_df.columns:
                adv_df["TOTAL_TIME_GLOBAL"] = adv_df["Time"] + total_time_so_far / 86400.0

            last_row = adv_df.tail(1)

            combined_df = pd.concat([combined_df, last_row], ignore_index=True)
 
            total_time_so_far = adv_df["TOTAL_TIME_GLOBAL"].iloc[-1] * 86400.0
            append_log(f"[{k:04d}] Δt={delta_t:.1f}s, Total={total_time_so_far:.1f}s "
                       f"({total_time_so_far / 86400:.3f} d)")

        combined_df.to_csv(COMBINED_CSV, index=False)


    # Drawing
    if not combined_df.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(combined_df["TOTAL_TIME_GLOBAL"], combined_df["U_mgL"], marker='o', linestyle='-')
        plt.xlabel("Time (day)")
        plt.ylabel("U concentration (mg/L)")
        plt.title("U concentration at core outlet vs global time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("全部循环完成。")
    append_log(f"# Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
