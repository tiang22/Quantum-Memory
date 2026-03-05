import numpy as np
import sinter
import itertools
from rotated_surface_code_circuit import get_rotated_surface_code_circuit_memory
from toric_code_circuit import get_toric_code_circuit_memory
from surface_code_circuit import get_surface_code_circuit_memory
from bivariate_bicycle_code_circuit import get_bivariate_bicycle_code_circuit_memory
import os
from stimbposd import SinterDecoder_BPOSD


def toric_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers,
    d_list,
    path,
    no_pc=False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
                if not no_pc:
                    yield sinter.Task(
                        circuit=get_toric_code_circuit_memory(
                            d,
                            3 * d,
                            tc,
                            0.1,
                            1.0,
                            5.0,
                            p,
                            p,
                            "full",
                            "ms_realistic",
                            basis=basis,
                        )[0],
                        decoder="pymatching",
                        json_metadata={
                            "d": d,
                            "r": 3 * d,
                            "p": p,
                            "mode": "all",
                            "pc": p,
                            "basis": basis,
                        },
                    )
                yield sinter.Task(
                    circuit=get_toric_code_circuit_memory(
                        d,
                        3 * d,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        p,
                        pc_fix if not no_pc else 0,
                        "full",
                        "ms_realistic",
                        basis=basis,
                    )[0],
                    decoder="pymatching",
                    json_metadata={
                        "d": d,
                        "r": 3 * d,
                        "p": p,
                        "mode": "fixed",
                        "pc": pc_fix if not no_pc else 0,
                        "basis": basis,
                    },
                )

    with open(path, "w") as f:
        print(sinter.CSV_HEADER, file=f)

        samples = sinter.collect(
            num_workers=num_workers,
            tasks=itertools.chain(generate_tasks("Z"), generate_tasks("X")),
            max_shots=max_shots,
            max_errors=max_errors,
            hint_num_tasks=len(p_list)
            * len(d_list)
            * (4 if not no_pc else 2),  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def surface_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers,
    d_list,
    path,
    no_pc=False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
                if not no_pc:
                    yield sinter.Task(
                        circuit=get_surface_code_circuit_memory(
                            d,
                            3 * d,
                            tc,
                            0.1,
                            1.0,
                            5.0,
                            p,
                            p,
                            "full",
                            "ms_realistic",
                            basis=basis,
                        )[0],
                        decoder="pymatching",
                        json_metadata={
                            "d": d,
                            "r": 3 * d,
                            "p": p,
                            "mode": "all",
                            "pc": p,
                            "basis": basis,
                        },
                    )
                yield sinter.Task(
                    circuit=get_surface_code_circuit_memory(
                        d,
                        3 * d,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        p,
                        pc_fix if not no_pc else 0,
                        "full",
                        "ms_realistic",
                        basis=basis,
                    )[0],
                    decoder="pymatching",
                    json_metadata={
                        "d": d,
                        "r": 3 * d,
                        "p": p,
                        "mode": "fixed",
                        "pc": pc_fix if not no_pc else 0,
                        "basis": basis,
                    },
                )

    with open(path, "w") as f:
        print(sinter.CSV_HEADER, file=f)

        samples = sinter.collect(
            num_workers=num_workers,
            tasks=itertools.chain(generate_tasks("Z"), generate_tasks("X")),
            max_shots=max_shots,
            max_errors=max_errors,
            hint_num_tasks=len(p_list)
            * len(d_list)
            * (4 if not no_pc else 2),  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def rotated_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers,
    d_list,
    path,
    no_pc=False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
                if not no_pc:
                    yield sinter.Task(
                        circuit=get_rotated_surface_code_circuit_memory(
                            d,
                            3 * d,
                            tc,
                            0.1,
                            1.0,
                            5.0,
                            p,
                            p,
                            "full",
                            "ms_realistic",
                            basis=basis,
                        )[0],
                        decoder="pymatching",
                        json_metadata={
                            "d": d,
                            "r": 3 * d,
                            "p": p,
                            "mode": "all",
                            "pc": p,
                            "basis": basis,
                        },
                    )
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        3 * d,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        p,
                        pc_fix if not no_pc else 0,
                        "full",
                        "ms_realistic",
                        basis=basis,
                    )[0],
                    decoder="pymatching",
                    json_metadata={
                        "d": d,
                        "r": 3 * d,
                        "p": p,
                        "mode": "fixed",
                        "pc": pc_fix if not no_pc else 0,
                        "basis": basis,
                    },
                )

    with open(path, "w") as f:
        print(sinter.CSV_HEADER, file=f)

        samples = sinter.collect(
            num_workers=num_workers,
            tasks=itertools.chain(generate_tasks("Z"), generate_tasks("X")),
            max_shots=max_shots,
            max_errors=max_errors,
            hint_num_tasks=len(p_list)
            * len(d_list)
            * (4 if not no_pc else 2),  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def bb_sim(
    l_list,
    m_list,
    a_lists,
    b_lists,
    d_list,
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers,
    path,
    no_pc=False,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    custom_bposd = SinterDecoder_BPOSD(max_bp_iters=50, osd_order=0)
    custom_decoders = {"bposd": custom_bposd}

    def generate_tasks(basis):
        for l, m, a_list, b_list, d in zip(
            l_list, m_list, a_lists, b_lists, d_list, strict=True
        ):
            for p in p_list:
                tc = -1 / np.log(1 - 4 / 3 * p)
                if not no_pc:
                    yield sinter.Task(
                        circuit=get_bivariate_bicycle_code_circuit_memory(
                            l,
                            m,
                            a_list,
                            b_list,
                            12,
                            tc,
                            0.1,
                            1.0,
                            5.0,
                            p,
                            p,
                            "full",
                            "ms_realistic",
                            basis=basis,
                        )[0],
                        decoder="bposd",
                        json_metadata={
                            "l": l,
                            "m": m,
                            "d": d,
                            "r": 12,
                            "p": p,
                            "mode": "all",
                            "pc": p,
                            "basis": basis,
                        },
                    )
                yield sinter.Task(
                    circuit=get_bivariate_bicycle_code_circuit_memory(
                        l,
                        m,
                        a_list,
                        b_list,
                        12,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        p,
                        pc_fix if not no_pc else 0,
                        "full",
                        "ms_realistic",
                        basis=basis,
                    )[0],
                    decoder="bposd",
                    json_metadata={
                        "l": l,
                        "m": m,
                        "d": d,
                        "r": 12,
                        "p": p,
                        "mode": "fixed",
                        "pc": pc_fix if not no_pc else 0,
                        "basis": basis,
                    },
                )

    with open(path, "w") as f:
        print(sinter.CSV_HEADER, file=f)

        samples = sinter.collect(
            num_workers=num_workers,
            tasks=itertools.chain(generate_tasks("Z"), generate_tasks("X")),
            max_shots=max_shots,
            max_errors=max_errors,
            hint_num_tasks=len(d_list)
            * len(p_list)
            * (4 if not no_pc else 2),  # 2 error configs x XZ
            decoders=["bposd"],
            custom_decoders=custom_decoders,
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


if __name__ == "__main__":
    """rotated_sim(
        10000000,
        10000,
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        10 ** (-4.5),
        16,
        d_list=[3, 5, 7, 9],
        path="results/rotated_sim.csv",
    )"""
    """toric_sim(
        10000000,
        10000,
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        10 ** (-4.5),
        16,
        d_list=[3, 5, 7],
        path="results/toric_sim.csv",
    )"""
    """surface_sim(
        10000000,
        10000,
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        10 ** (-4.5),
        16,
        d_list=[3, 5, 7],
        path="results/surface_sim.csv",
    )"""
    bb_sim(
        [6],
        [6],
        [[3, 1, 2]],
        [[3, 1, 2]],
        [6],
        50000,
        100,
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        10 ** (-4.5),
        64,
        path="results/bb_sim_no_pc_72_12_6_50000_100.csv",
        no_pc=True,
    )
