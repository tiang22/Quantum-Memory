import numpy as np
import sinter
import itertools
from rotated_surface_code_circuit import get_rotated_surface_code_circuit_memory
from toric_code_circuit import get_toric_code_circuit_memory
from surface_code_circuit import get_surface_code_circuit_memory
from bivariate_bicycle_code_circuit import get_bivariate_bicycle_code_circuit_memory
import os
from stimbposd import SinterDecoder_BPOSD


def rotated_sim_single_error(
    max_shots,
    max_errors,
    ds,
    error_list,
    cnot_scheduling="full",
    num_workers=16,
    crosstalk_noise="ms_realistic",
    path="results/rotated_single_error.csv",
):
    def generate_tasks(basis):
        for d in ds:
            for err in error_list:
                tc = -1 / np.log(1 - 4 / 3 * err)
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        0.0,
                        0.0,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "idle",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        0.0,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "infidelity",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "crosstalk",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        err,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "all",
                        "p": err,
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
            decoders=["pymatching"],
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def toric_sim_single_error(
    max_shots,
    max_errors,
    ds,
    error_list,
    cnot_scheduling="full",
    num_workers=16,
    crosstalk_noise="ms_realistic",
    path="results/toric_single_error.csv",
):
    def generate_tasks(basis):
        for d in ds:
            for err in error_list:
                tc = -1 / np.log(1 - 4 / 3 * err)
                yield sinter.Task(
                    circuit=get_toric_code_circuit_memory(
                        d,
                        d * 3,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        0.0,
                        0.0,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "idle",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_toric_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        0.0,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "infidelity",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_toric_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "crosstalk",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_toric_code_circuit_memory(
                        d,
                        d * 3,
                        tc,
                        0.1,
                        1.0,
                        5.0,
                        err,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "noise_type": "all",
                        "p": err,
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
            decoders=["pymatching"],
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def rotated_sim_scheduling_coherence(
    max_shots,
    max_errors,
    d,
    rounds,
    coherent_ratio_list,
    gate_1_ratio,
    measure_ratio,
    infidelity,
    crosstalk_list,
    cnot_scheduling_list,
    crosstalk_noise,
    data_path,
    num_workers=64,
    pad_time=False,
):
    # qubit coherent time: 1e-1 - 1e+0
    # two qubit gate time: 1e-4 - 5e-4
    # one qubit gate time: 1e-5 - 5e-5 (1/10 of two qubit gate time)
    # measure time: 5e-4 - 1e-3

    # coherent_ratio: coherent time / two qubit gate time
    # gate_1_ratio: single qubit gate time / two qubit gate time
    # measure_ratio: measure time / two qubit gate time

    def generate_tasks(basis):
        # reference serial
        time_serial = 0.0
        for tc in coherent_ratio_list:
            circuit, time = get_rotated_surface_code_circuit_memory(
                d=d,
                rounds=rounds,
                qubit_coherent_time=tc,
                gate_1_time=gate_1_ratio,
                gate_2_time=1.0,
                measure_time=measure_ratio,
                infidelity=infidelity,
                crosstalk=0.0,
                cnot_scheduling="serial",
                crosstalk_noise=crosstalk_noise,
                basis=basis,
            )
            yield sinter.Task(
                circuit=circuit,
                json_metadata={
                    "d": d,
                    "r": rounds,
                    "t": time,
                    "group": "serial",
                    "tc": tc,
                    "tg1": gate_1_ratio,
                    "tg2": 1.0,
                    "tm": measure_ratio,
                    "infidelity": infidelity,
                    "crosstalk": 0.0,
                    "cnot_scheduling": "serial",
                    "crosstalk_noise": crosstalk_noise,
                    "basis": basis,
                },
            )
            time_serial = time

        # parallel
        for cnot_scheduling in cnot_scheduling_list:
            for tc in coherent_ratio_list:
                for i, crosstalk in enumerate(crosstalk_list):
                    circuit, time = get_rotated_surface_code_circuit_memory(
                        d=d,
                        rounds=rounds,
                        qubit_coherent_time=tc,
                        gate_1_time=gate_1_ratio,
                        gate_2_time=1.0,
                        measure_time=measure_ratio,
                        infidelity=infidelity,
                        crosstalk=crosstalk,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        basis=basis,
                    )
                    if pad_time:
                        circuit, time = get_rotated_surface_code_circuit_memory(
                            d=d,
                            rounds=rounds,
                            qubit_coherent_time=tc,
                            gate_1_time=gate_1_ratio,
                            gate_2_time=1.0,
                            measure_time=measure_ratio,
                            infidelity=infidelity,
                            crosstalk=crosstalk,
                            cnot_scheduling=cnot_scheduling,
                            crosstalk_noise=crosstalk_noise,
                            pad_time_per_cycle=(time_serial - time) / rounds,
                            basis=basis,
                        )
                    yield sinter.Task(
                        circuit=circuit,
                        json_metadata={
                            "d": d,
                            "r": rounds,
                            "t": time,
                            "group": i,
                            "tc": tc,
                            "tg1": gate_1_ratio,
                            "tg2": 1.0,
                            "tm": measure_ratio,
                            "infidelity": infidelity,
                            "crosstalk": crosstalk,
                            "cnot_scheduling": cnot_scheduling,
                            "crosstalk_noise": crosstalk_noise,
                            "basis": basis,
                        },
                    )
                # no crosstalk
                circuit, time = get_rotated_surface_code_circuit_memory(
                    d=d,
                    rounds=rounds,
                    qubit_coherent_time=tc,
                    gate_1_time=gate_1_ratio,
                    gate_2_time=1.0,
                    measure_time=measure_ratio,
                    infidelity=infidelity,
                    crosstalk=0.0,
                    cnot_scheduling=cnot_scheduling,
                    crosstalk_noise=crosstalk_noise,
                    basis=basis,
                )
                if pad_time:
                    circuit, time = get_rotated_surface_code_circuit_memory(
                        d=d,
                        rounds=rounds,
                        qubit_coherent_time=tc,
                        gate_1_time=gate_1_ratio,
                        gate_2_time=1.0,
                        measure_time=measure_ratio,
                        infidelity=infidelity,
                        crosstalk=0.0,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise=crosstalk_noise,
                        pad_time_per_cycle=(time_serial - time) / rounds,
                        basis=basis,
                    )
                yield sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "d": d,
                        "r": rounds,
                        "t": time,
                        "group": -1,
                        "tc": tc,
                        "tg1": gate_1_ratio,
                        "tg2": 1.0,
                        "tm": measure_ratio,
                        "infidelity": infidelity,
                        "crosstalk": 0.0,
                        "cnot_scheduling": cnot_scheduling,
                        "crosstalk_noise": crosstalk_noise,
                        "basis": basis,
                    },
                )

    with open(data_path, "w") as f:
        print(sinter.CSV_HEADER, file=f)
        samples = sinter.collect(
            num_workers=num_workers,
            tasks=itertools.chain(generate_tasks("Z"), generate_tasks("X")),
            decoders=["pymatching"],
            max_shots=max_shots,
            max_errors=max_errors,
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def toric_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers=16,
    d_list=[3, 5, 7],
    path="results/toric_sim.csv",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
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
                        pc_fix,
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
                        "pc": pc_fix,
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
            hint_num_tasks=len(p_list) * len(d_list) * 4,  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def rotated_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers=16,
    d_list=[3, 5, 7],
    path="results/rotated_sim.csv",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
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
                        pc_fix,
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
                        "pc": pc_fix,
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
            hint_num_tasks=len(p_list) * len(d_list) * 4,  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def surface_sim(
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers=16,
    d_list=[3, 5, 7],
    path="results/surface_sim.csv",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
            for d in d_list:
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
                        pc_fix,
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
                        "pc": pc_fix,
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
            hint_num_tasks=len(p_list) * len(d_list) * 4,  # 2 error configs x XZ
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def bb_sim(
    l,
    m,
    a_list,
    b_list,
    max_shots,
    max_errors,
    p_list,
    pc_fix,
    num_workers=16,
    path="results/bb_sim.csv",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    custom_bposd = SinterDecoder_BPOSD(osd_order=0)
    custom_decoders = {"bposd": custom_bposd}

    def generate_tasks(basis):
        for p in p_list:
            tc = -1 / np.log(1 - 4 / 3 * p)
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
                    pc_fix,
                    "full",
                    "ms_realistic",
                    basis=basis,
                )[0],
                decoder="bposd",
                json_metadata={
                    "l": l,
                    "m": m,
                    "r": 12,
                    "p": p,
                    "mode": "all",
                    "pc": pc_fix,
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
            hint_num_tasks=len(p_list) * 4,  # 2 error configs x XZ
            decoders=["bposd"],
            custom_decoders=custom_decoders,
            print_progress=True,
        )
        for sample in samples:
            print(sample.to_csv_line(), file=f)


def rotated_sim_crosstalk_noise(
    max_shots,
    max_errors,
    ds,
    error_list,
    cnot_scheduling="full",
    num_workers=16,
    path="results/rotated_sim_crosstalk_noise.csv",
):
    def generate_tasks(basis):
        for d in ds:
            for err in error_list:
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise="depolarize",
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "crosstalk_noise": "depolarize",
                        "p": err,
                        "basis": basis,
                    },
                )
                yield sinter.Task(
                    circuit=get_rotated_surface_code_circuit_memory(
                        d,
                        d * 3,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        err,
                        cnot_scheduling=cnot_scheduling,
                        crosstalk_noise="ms_realistic",
                        basis=basis,
                    )[0],
                    json_metadata={
                        "d": d,
                        "r": d * 3,
                        "crosstalk_noise": "ms_realistic",
                        "p": err,
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
            decoders=["pymatching"],
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
        6,
        6,
        [3, 1, 2],
        [3, 1, 2],
        100000,
        100,
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        10 ** (-4.5),
        8,
        path="results/bb_sim.csv",
    )
    """rotated_sim_single_error(
        10000000,
        100000,
        [3, 5, 7, 9, 11],
        np.power(10.0, np.linspace(-6.0, -1.0, num=26)).tolist(),
        "full",
        64,
        path="results/rotated_single_error.csv",
    )
    toric_sim_single_error(
        10000000,
        100000,
        [3, 5, 7, 9, 11],
        np.power(10.0, np.linspace(-6.0, -1.0, num=26)).tolist(),
        "full",
        64,
        path="results/toric_single_error.csv",
    )
    rotated_sim_scheduling_coherence(
        10000000,
        100000,
        5,
        15,
        np.power(10.0, np.linspace(2.0, 5.0, num=31)).tolist(),
        0.1,
        5.0,
        1e-3,
        np.power(10.0, np.linspace(-7.0, -4.0, num=4)).tolist(),
        ["full"],
        "ms_realistic",
        "results/rotated_sim_schedule_coherence_coherence_crosstalk.csv",
        64,
    )
    rotated_sim_scheduling_coherence(
        10000000,
        100000,
        5,
        15,
        np.power(10.0, np.linspace(2.0, 5.0, num=31)).tolist(),
        0.1,
        5.0,
        1e-3,
        [1e-5],
        ["full", "half", "sqrt", "minimal"],
        "ms_realistic",
        "results/rotated_sim_schedule_coherence_coherence_schedule.csv",
        64,
    )
    rotated_sim_scheduling_coherence(
        10000000,
        100000,
        5,
        15,
        np.power(10.0, np.linspace(2.0, 5.0, num=31)).tolist(),
        0.1,
        5.0,
        1e-3,
        [1e-5],
        ["full", "half", "sqrt", "minimal"],
        "ms_realistic",
        "results/rotated_sim_schedule_coherence_coherence_schedule_pad.csv",
        64,
        True,
    )
    rotated_sim_scheduling_coherence(
        10000000,
        100000,
        5,
        15,
        [10**3.5],
        0.1,
        5.0,
        1e-3,
        np.power(10.0, np.linspace(-7.0, -3.0, num=21)).tolist(),
        ["full", "half", "sqrt", "minimal"],
        "ms_realistic",
        "results/rotated_sim_schedule_coherence_crosstalk_schedule.csv",
        64,
    )
    rotated_sim_scheduling_coherence(
        10000000,
        100000,
        5,
        15,
        [10**3.5],
        0.1,
        5.0,
        1e-3,
        np.power(10.0, np.linspace(-7.0, -3.0, num=21)).tolist(),
        ["full", "half", "sqrt", "minimal"],
        "ms_realistic",
        "results/rotated_sim_schedule_coherence_crosstalk_schedule_pad.csv",
        64,
        True,
    )
    rotated_sim_crosstalk_noise(
        100000000,
        100000,
        [5],
        np.power(10.0, np.linspace(-7.0, -1.0, num=31)).tolist(),
        "full",
        64,
        path="results/rotated_sim_crosstalk_noise.csv",
    )"""
