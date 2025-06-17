import re
import pandas as pd

def parse_log(text):
    # Task行を抽出
    task_pattern = r"Task\s+AppTime\s+MPITime\s+MPI%(.+?)\n\s*\*"
    task_block = re.search(task_pattern, text, re.DOTALL)
    tasks = []
    if task_block:
        lines = task_block.group(1).strip().split('\n')
        for line in lines:
            cols = line.split()
            if len(cols) == 4:
                task_id, apptime, mpitime, mpi_perc = cols
                tasks.append({
                    'Task': task_id,
                    'AppTime': float(apptime),
                    'MPITime': float(mpitime),
                    'MPI%': float(mpi_perc)
                })
    df_tasks = pd.DataFrame(tasks)

    # Callsitesの抽出
    callsite_pattern = r"@--- Callsites: \d+ -+\n(.+?)\n-+\n"
    callsite_block = re.search(callsite_pattern, text, re.DOTALL)
    callsites = []
    if callsite_block:
        lines = callsite_block.group(1).strip().split('\n')
        for line in lines:
            # 例:  1   0 0x766a74e173fa           [unknown]                Gather
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 7:
                ID, Lev, FileAddr, Line, ParentFunct, *MPI_Call = parts
                mpi_call_str = ' '.join(MPI_Call)
                callsites.append({
                    'ID': int(ID),
                    'Lev': int(Lev),
                    'File/Address': FileAddr,
                    'Line': Line,
                    'Parent_Funct': ParentFunct,
                    'MPI_Call': mpi_call_str
                })
    df_callsites = pd.DataFrame(callsites)

    # Aggregate Time抽出
    agg_time_pattern = r"@--- Aggregate Time \(top twenty.+?\n-+\n(.+?)\n-+\n"
    agg_time_block = re.search(agg_time_pattern, text, re.DOTALL)
    agg_times = []
    if agg_time_block:
        lines = agg_time_block.group(1).strip().split('\n')
        for line in lines:
            parts = re.split(r'\s+', line.strip())
            # ヘッダー行や空行をスキップ
            if not parts or parts[0] == "Call" or parts[0].startswith('-'):
                continue
            if len(parts) >= 7:
                call, site, time_ms, app_perc, mpi_perc, count, cov = parts[:7]
                agg_times.append({
                    'Call': call,
                    'Site': int(site),
                    'Time(ms)': float(time_ms),
                    'App%': float(app_perc),
                    'MPI%': float(mpi_perc),
                    'Count': int(count),
                    'COV': float(cov)
                })
    df_agg_times = pd.DataFrame(agg_times)

    return df_tasks, df_callsites, df_agg_times


# 使用例
if __name__ == "__main__":
    with open("cnn_mpi.4.26813.1.txt", "r") as f:
        log_text = f.read()

    df_tasks, df_callsites, df_agg_times = parse_log(log_text)

    print("=== Task Data ===")
    print(df_tasks)
    print("\n=== Callsites ===")
    print(df_callsites.head())
    print("\n=== Aggregate Times ===")
    print(df_agg_times.head())

