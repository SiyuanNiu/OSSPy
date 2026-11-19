import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sys

phreeqc_exe = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\Release\phreeqc.exe"
PHREEQC_DB = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\phreeqc.dat"

EX_INPUT = "ex_step.pqi"       # Initial Reaction 
TRANS_INPUT = "trans_step.pqi"     # Initial Transport 
EX_STDOUT = "ex_out.txt"
TRANS_STDOUT = "trans_out.txt"

# Output Files
COMBINED_LOG = "ex11_operator_splitting.log"
COMBINED_CSV = "ex11_all_selected.csv"

N_CYCLES = 100   
PRINT_EVERY = 20


def run_once(exe, in_file, std_file, db_file):
    cmd = [exe, in_file, std_file, db_file]
    ret = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return ret.returncode, ret.stdout, ret.stderr

def append_log(msg):
    with open(COMBINED_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def read_sel_file(path):
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#')
        return df
    except Exception:
        return pd.DataFrame()

def main():
    for f in (EX_INPUT, TRANS_INPUT):
        if not os.path.exists(f):
            print(f"[ERROR] 找不到 {f}")
            sys.exit(1)

    if os.path.exists(COMBINED_CSV):
        os.remove(COMBINED_CSV)
    combined_df = pd.DataFrame()

    with open(COMBINED_LOG, "w", encoding="utf-8") as f:
        f.write(f"# ex11_operator_splitting started at {datetime.now()}\n")

    total_time_so_far = 0.0
    last_ex_mtime = None
    last_trans_mtime = None

    cell_no = 40
    global_step = 0



    for k in range(1, N_CYCLES + 1):
        # --- EM step ---
        if k == 1:
            ex_input_file = EX_INPUT
        else:
            with open(EX_INPUT, "r", encoding="utf-8") as f:
                ex_lines = f.readlines()

            new_lines = []
            inserted = False
            skip_solution = False

            for i, line in enumerate(ex_lines):
                low = line.strip().lower()

                if low.startswith("solution"):
                    skip_solution = True
                    continue

                if skip_solution:
                    if any(low.startswith(x) for x in ["copy", "dump", "selected_output"]):
                        skip_solution = False
                    else:
                        continue 
                if not inserted and low.startswith("copy"):
                    new_lines.append("INCLUDE$ trans_solutions.pqi\n")
                    inserted = True

                new_lines.append(line)

            while new_lines and new_lines[-1].strip().upper() == "END":
                new_lines = new_lines[:-1]
            new_lines.append("END\n")

            ex_input_file = "ex_step_tmp.pqi"
            with open(ex_input_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        rc, out, err = run_once(phreeqc_exe, ex_input_file, EX_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[EX FAIL] {k}, rc={rc}\n{err}")
            sys.exit(rc)
        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] EX step OK")
        append_log(f"[{k}/{N_CYCLES}] EX step OK")

        if os.path.exists("ex_solutions.pqi"):
            mtime = os.path.getmtime("ex_solutions.pqi")
            if last_ex_mtime is not None and mtime == last_ex_mtime:
                raise RuntimeError(f"[Check FAIL] EX step {k} 没有更新 ex_solutions.pqi (mtime 未变)")
            print(f"[Check PASS] EX step {k} 更新 ex_solutions.pqi (mtime={time.ctime(mtime)})")
            last_ex_mtime = mtime


        # --- Transport step ---
        rc, out, err = run_once(phreeqc_exe, TRANS_INPUT, TRANS_STDOUT, PHREEQC_DB)
        if rc != 0:
            print(f"[TRANS FAIL] {k}, rc={rc}\n{err}")
            sys.exit(rc)
        if k % PRINT_EVERY == 0 or k == 1:
            print(f"[{k}/{N_CYCLES}] TRANS step OK")
        append_log(f"[{k}/{N_CYCLES}] TRANS step OK")

        if os.path.exists("trans_solutions.pqi"):
            mtime = os.path.getmtime("trans_solutions.pqi")
            if last_trans_mtime is not None and mtime == last_trans_mtime:
                raise RuntimeError(f"[Check FAIL] TRANS step {k} 没有更新 trans_solutions.pqi (mtime 未变)")
            print(f"[Check PASS] TRANS step {k} 更新 trans_solutions.pqi (mtime={time.ctime(mtime)})")
            last_trans_mtime = mtime

        # Results
        trans_df = read_sel_file("trans_selected_out.sel")
  
        if not trans_df.empty:
            
            trans_df['step'] = pd.to_numeric(trans_df['step'], errors='coerce').fillna(0).astype(int)

            local_min = int(trans_df['step'].min())
            local_max = int(trans_df['step'].max())

            steps_advanced = max(local_max - local_min, 1)
            global_step += steps_advanced

            last_row = trans_df[trans_df['step'] == local_max].tail(1).copy()

            last_row['Pore_volume'] = (global_step + 0.5) / cell_no

            for ion in ["Na", "Cl", "K", "Ca"]:
                if ion in last_row.columns:
                    last_row[ion + "_mM"] = last_row[ion] * 1000

            combined_df = pd.concat([combined_df, last_row], ignore_index=True)

        combined_df.to_csv(COMBINED_CSV, index=False)


    # Drawing
    if not combined_df.empty:
        plt.figure(figsize=(7, 4))
        for ion in ["Na", "Cl", "K", "Ca"]:
            col = ion + "_mM"
            if col in combined_df.columns:
                plt.plot(combined_df["Pore_volume"], combined_df[col], label=ion)
        plt.xlabel("Pore volumes")
        plt.ylabel("mmol/kgw")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
