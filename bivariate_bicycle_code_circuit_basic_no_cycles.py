import stim
import numpy as np
import random
from qldpc.codes import CSSCode


def get_bivariate_bicycle_code_circuit_memory_basic_no_cycles(
    l,  # subsystem A size
    m,  # subsystem B size
    a_list,
    b_list,
    rounds,
    qubit_coherent_time,
    gate_1_time,
    gate_2_time,
    measure_time,
    infidelity,
    crosstalk,
    cnot_scheduling="full",
    crosstalk_noise="depolarize",
    pad_time_per_cycle=0.0,
    basis="Z",
):
    assert len(a_list) == len(b_list) == 3
    circuit = stim.Circuit()
    syn_ext = stim.Circuit()
    no_error_cycle = stim.Circuit()
    n = 2 * m * l  # code length
    n2 = m * l
    num_X_checks = m * l
    num_Z_checks = m * l

    I_l = np.identity(l, dtype=int)
    I_m = np.identity(m, dtype=int)

    x = {}
    y = {}
    for i in range(l):
        x[i] = np.kron(np.roll(I_l, i, axis=1), I_m)
    for i in range(m):
        y[i] = np.kron(I_l, np.roll(I_m, i, axis=1))

    A = (x[a_list[0]] + y[a_list[1]] + y[a_list[2]]) % 2
    B = (y[b_list[0]] + x[b_list[1]] + x[b_list[2]]) % 2

    A1 = x[a_list[0]]
    A2 = y[a_list[1]]
    A3 = y[a_list[2]]
    B1 = y[b_list[0]]
    B2 = x[b_list[1]]
    B3 = x[b_list[2]]

    X_check_X_offset_2_data = {}  # (check_index, offset) -> data_index
    Z_check_Z_offset_2_data = {}

    for X_check in range(n, n + num_X_checks):
        X_check_X_offset_2_data[(X_check, 0)] = np.nonzero(A1[X_check - n, :])[0][0]
        X_check_X_offset_2_data[(X_check, 1)] = np.nonzero(A2[X_check - n, :])[0][0]
        X_check_X_offset_2_data[(X_check, 2)] = np.nonzero(A3[X_check - n, :])[0][0]
        X_check_X_offset_2_data[(X_check, 3)] = (
            np.nonzero(B1[X_check - n, :])[0][0] + n2
        )
        X_check_X_offset_2_data[(X_check, 4)] = (
            np.nonzero(B2[X_check - n, :])[0][0] + n2
        )
        X_check_X_offset_2_data[(X_check, 5)] = (
            np.nonzero(B3[X_check - n, :])[0][0] + n2
        )

    for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
        Z_check_Z_offset_2_data[(Z_check, 0)] = np.nonzero(
            B1[:, Z_check - n - num_X_checks]
        )[0][0]
        Z_check_Z_offset_2_data[(Z_check, 1)] = np.nonzero(
            B2[:, Z_check - n - num_X_checks]
        )[0][0]
        Z_check_Z_offset_2_data[(Z_check, 2)] = np.nonzero(
            B3[:, Z_check - n - num_X_checks]
        )[0][0]
        Z_check_Z_offset_2_data[(Z_check, 3)] = (
            np.nonzero(A1[:, Z_check - n - num_X_checks])[0][0] + n2
        )
        Z_check_Z_offset_2_data[(Z_check, 4)] = (
            np.nonzero(A2[:, Z_check - n - num_X_checks])[0][0] + n2
        )
        Z_check_Z_offset_2_data[(Z_check, 5)] = (
            np.nonzero(A3[:, Z_check - n - num_X_checks])[0][0] + n2
        )

    AT = np.transpose(A)
    BT = np.transpose(B)

    HX = np.hstack((A, B))
    HZ = np.hstack((BT, AT))

    code = CSSCode(code_x=HX.astype(int), code_z=HZ.astype(int))
    logicals = code.get_logical_ops()
    k2 = logicals.shape[0]
    assert logicals.shape[1] == 2 * n
    X_logicals = logicals[: k2 // 2, :n]
    Z_logicals = logicals[k2 // 2 :, n:]

    X_neighbors = ["idle", 1, 4, 3, 5, 0, 2]
    Z_neighbors = [3, 5, 0, 1, 2, 4, "idle"]

    time = 0.0
    time_cycle = 0.0

    def add_idle_noise(circuit, targets, tick_length):
        if tick_length > 0:
            circuit.append(
                "DEPOLARIZE1",
                targets,
                0.75 * (1 - np.exp(-tick_length / qubit_coherent_time)),
            )

    def add_X_noise(circuit, targets, tick_length):
        if tick_length > 0:
            circuit.append(
                "X_ERROR",
                targets,
                0.75 * (1 - np.exp(-tick_length / qubit_coherent_time)),
            )

    def add_Z_noise(circuit, targets, tick_length):
        if tick_length > 0:
            circuit.append(
                "Z_ERROR",
                targets,
                0.75 * (1 - np.exp(-tick_length / qubit_coherent_time)),
            )

    def add_infidelity(circuit, targets):
        if infidelity > 0:
            circuit.append("DEPOLARIZE2", targets, infidelity)

    def add_crosstalk(circuit, targets):
        if crosstalk > 0:
            q1 = [
                targets[i] for i in range(0, len(targets), 2)
            ]  # even indices are control qubits
            q2 = [
                targets[i] for i in range(1, len(targets), 2)
            ]  # odd indices are target qubits

            t11 = []
            t12 = []
            t22 = []
            for i in range(len(q1)):
                for j in range(i + 1, len(q1)):
                    t11.extend([q1[i], q1[j]])
                for j in range(len(q2)):
                    if i != j:  # no crosstalk within a single CNOT
                        t12.extend([q1[i], q2[j]])
            for i in range(len(q2)):
                for j in range(i + 1, len(q2)):
                    t22.extend([q2[i], q2[j]])
            if crosstalk_noise == "depolarize":
                circuit.append("DEPOLARIZE2", t11 + t12 + t22, crosstalk)
            elif crosstalk_noise == "ms_realistic":
                for i in range(0, len(t11), 2):
                    circuit.append_from_stim_program_text(
                        f"PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,{crosstalk}) {t11[i]} {t11[i+1]}"
                    )
                for i in range(0, len(t12), 2):
                    circuit.append_from_stim_program_text(
                        f"PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,{crosstalk},0,0) {t12[i]} {t12[i+1]}"
                    )
                for i in range(0, len(t22), 2):
                    circuit.append_from_stim_program_text(
                        f"PAULI_CHANNEL_2(0,0,0,0,{crosstalk},0,0,0,0,0,0,0,0,0,0) {t22[i]} {t22[i+1]}"
                    )
            else:
                raise NotImplementedError

    def add_cnot(circuit, all_qubits, cnot_targets):
        circuit.append("CX", cnot_targets)
        add_idle_noise(
            circuit, list(set(all_qubits).difference(set(cnot_targets))), gate_2_time
        )
        add_infidelity(circuit, cnot_targets)
        add_crosstalk(circuit, cnot_targets)
        circuit.append("TICK")

    for i in range(n + num_X_checks + num_Z_checks):
        # data X Z order
        circuit.append_from_stim_program_text(f"QUBIT_COORDS({i}) {i}")

    if basis == "Z":
        circuit.append(
            "R",
            list(range(n + num_X_checks + num_Z_checks)),
        )
    elif basis == "X":
        circuit.append("RX", list(range(n)))
        circuit.append("R", list(range(n, n + num_X_checks + num_Z_checks)))
    add_idle_noise(
        circuit,
        list(range(n + num_X_checks + num_Z_checks)),
        0.0,
    )
    circuit.append("TICK")
    circuit.append("H", list(range(n, n + num_X_checks)))
    """add_idle_noise(
        circuit,
        list(range(n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )"""
    circuit.append("TICK")

    syn_ext.append("H", list(range(n, n + num_X_checks)))
    """add_X_noise(
        syn_ext,
        list(
            range(n + num_X_checks, n + num_X_checks + num_Z_checks)
        ),  # preparation noise on Z checks
        gate_1_time,
    )"""
    syn_ext.append("TICK")
    no_error_cycle.append("H", list(range(n, n + num_X_checks)))
    no_error_cycle.append("TICK")

    time += gate_1_time
    time_cycle += gate_1_time

    targets_CX = [{} for _ in range(len(X_neighbors))]
    for t, (X_offset, Z_offset) in enumerate(
        zip(X_neighbors, Z_neighbors, strict=True)
    ):
        if X_offset != "idle":
            for X_check in range(n, n + num_X_checks):
                data = X_check_X_offset_2_data[(X_check, X_offset)]
                targets_CX[t][X_check] = (data, True)
        if Z_offset != "idle":
            for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
                data = Z_check_Z_offset_2_data[(Z_check, Z_offset)]
                targets_CX[t][Z_check] = (data, False)

    def generate_cnot_scheduling(cnot_scheduling):
        targets_CX_tick = []
        if cnot_scheduling == "full":
            for t in range(len(X_neighbors)):
                targets_CX_tick.append([])
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        targets_CX_tick[t].extend([anc, data])
                    else:
                        targets_CX_tick[t].extend([data, anc])
        elif cnot_scheduling == "half":
            for t in range(len(X_neighbors)):
                targets_CX_tick.extend([[], []])
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        targets_CX_tick[t * 2].extend([anc, data])
                    else:
                        targets_CX_tick[t * 2 + 1].extend([data, anc])
        elif cnot_scheduling == "minimal":
            for t in range(len(X_neighbors)):
                anc_list = list(targets_CX[t].keys())
                random.shuffle(anc_list)
                for i in range(0, len(anc_list), 2):
                    targets_this_tick = []
                    data_1, is_X_1 = targets_CX[t][anc_list[i]]
                    data_2, is_X_2 = targets_CX[t][anc_list[i + 1]]
                    if is_X_1:
                        targets_this_tick.extend([anc_list[i], data_1])
                    else:
                        targets_this_tick.extend([data_1, anc_list[i]])
                    if is_X_2:
                        targets_this_tick.extend([anc_list[i + 1], data_2])
                    else:
                        targets_this_tick.extend([data_2, anc_list[i + 1]])
                    targets_CX_tick.append(targets_this_tick)
        elif cnot_scheduling == "serial":
            for t in range(len(X_neighbors)):
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        targets_CX_tick.append([anc, data])
                    else:
                        targets_CX_tick.append([data, anc])
        else:
            raise NotImplementedError
        return targets_CX_tick

    targets_CX_tick = generate_cnot_scheduling(
        cnot_scheduling
    )  # generate CNOT scheduling
    add_Z_noise(
        circuit,
        list(range(n, n + num_X_checks)),  # preparation noise on X checks
        gate_2_time,
    )
    add_Z_noise(
        syn_ext,
        list(range(n, n + num_X_checks)),
        gate_2_time,
    )
    for cnot_targets in targets_CX_tick:
        add_cnot(
            circuit,
            list(range(n)),  # n for only data qubits
            cnot_targets,
        )
        add_cnot(
            syn_ext,
            list(range(n)),
            cnot_targets,
        )
        time += gate_2_time
        time_cycle += gate_2_time
        add_idle_noise(
            circuit,
            list(range(n + num_X_checks + num_Z_checks)),
            pad_time_per_cycle / len(targets_CX_tick),
        )
        add_idle_noise(
            syn_ext,
            list(range(n + num_X_checks + num_Z_checks)),
            pad_time_per_cycle / len(targets_CX_tick),
        )
        time += pad_time_per_cycle / len(targets_CX_tick)
        time_cycle += pad_time_per_cycle / len(targets_CX_tick)
        no_error_cycle.append("CX", cnot_targets)
        no_error_cycle.append("TICK")

    circuit.append("H", range(n, n + num_X_checks))
    """add_idle_noise(
        circuit,
        list(range(n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )"""
    circuit.append("TICK")
    add_idle_noise(
        circuit,
        list(range(n)),
        measure_time,
    )
    add_X_noise(
        circuit,
        list(
            range(n, n + num_X_checks + num_Z_checks)
        ),  # measurement error on X, Z checks
        measure_time,
    )
    if basis == "Z":
        circuit.append("MR", list(range(n, n + num_X_checks + num_Z_checks)))
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            circuit.append_from_stim_program_text(
                f"DETECTOR({Z_check}, 0) rec[{Z_check - n - num_X_checks - num_Z_checks}]"
            )
    elif basis == "X":
        circuit.append(
            "MR",
            list(range(n + num_X_checks, n + num_X_checks + num_Z_checks))
            + list(range(n, n + num_X_checks)),
        )
        for X_check in range(n, n + num_X_checks):
            circuit.append_from_stim_program_text(
                f"DETECTOR({X_check}, 0) rec[{X_check - n - num_X_checks}]"
            )
    else:
        raise NotImplementedError
    add_X_noise(
        circuit,
        list(range(n + num_X_checks, n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )  # preparation noise on Z checks
    circuit.append("TICK")

    syn_ext.append("H", list(range(n, n + num_X_checks)))
    """add_idle_noise(
        syn_ext,
        list(range(n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )"""
    syn_ext.append("TICK")
    add_idle_noise(
        syn_ext,
        list(range(n)),
        measure_time,
    )
    add_X_noise(
        syn_ext,
        list(
            range(n, n + num_X_checks + num_Z_checks)
        ),  # measurement error on X, Z checks
        measure_time,
    )
    if basis == "Z":
        syn_ext.append("MR", list(range(n, n + num_X_checks + num_Z_checks)))
        syn_ext.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
        for X_check in range(n, n + num_X_checks):
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({X_check}, 0) rec[{X_check - n - 2 * num_X_checks - 2 * num_Z_checks}] rec[{X_check - n - num_X_checks - num_Z_checks}]"
            )
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({Z_check}, 0) rec[{Z_check - n - num_X_checks - 2 * num_Z_checks - num_X_checks}] rec[{Z_check - n - num_X_checks - num_Z_checks}]"
            )
    elif basis == "X":
        syn_ext.append(
            "MR",
            list(range(n + num_X_checks, n + num_X_checks + num_Z_checks))
            + list(range(n, n + num_X_checks)),
        )
        syn_ext.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({Z_check}, 0) rec[{Z_check - n - num_X_checks - 2 * num_Z_checks - 2 * num_X_checks}] rec[{Z_check - n - num_X_checks - num_Z_checks - num_X_checks}]"
            )
        for X_check in range(n, n + num_X_checks):
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({X_check}, 0) rec[{X_check - n - 2 * num_X_checks - num_Z_checks}] rec[{X_check - n - num_X_checks}]"
            )
    else:
        raise NotImplementedError
    add_X_noise(
        syn_ext,
        list(range(n + num_X_checks, n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )  # preparation noise on Z checks
    syn_ext.append("TICK")

    no_error_cycle.append("H", list(range(n, n + num_X_checks)))
    """add_idle_noise(
        syn_ext,
        list(range(n + num_X_checks + num_Z_checks)),
        gate_1_time,
    )"""
    no_error_cycle.append("TICK")
    if basis == "Z":
        no_error_cycle.append("MR", list(range(n, n + num_X_checks + num_Z_checks)))
        no_error_cycle.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
        for X_check in range(n, n + num_X_checks):
            no_error_cycle.append_from_stim_program_text(
                f"DETECTOR({X_check}, 0) rec[{X_check - n - 2 * num_X_checks - 2 * num_Z_checks}] rec[{X_check - n - num_X_checks - num_Z_checks}]"
            )
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            no_error_cycle.append_from_stim_program_text(
                f"DETECTOR({Z_check}, 0) rec[{Z_check - n - num_X_checks - 2 * num_Z_checks - num_X_checks}] rec[{Z_check - n - num_X_checks - num_Z_checks}]"
            )
    elif basis == "X":
        no_error_cycle.append(
            "MR",
            list(range(n + num_X_checks, n + num_X_checks + num_Z_checks))
            + list(range(n, n + num_X_checks)),
        )
        no_error_cycle.append_from_stim_program_text("SHIFT_COORDS(0, 1)")
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            no_error_cycle.append_from_stim_program_text(
                f"DETECTOR({Z_check}, 0) rec[{Z_check - n - num_X_checks - 2 * num_Z_checks - 2 * num_X_checks}] rec[{Z_check - n - num_X_checks - num_Z_checks - num_X_checks}]"
            )
        for X_check in range(n, n + num_X_checks):
            no_error_cycle.append_from_stim_program_text(
                f"DETECTOR({X_check}, 0) rec[{X_check - n - 2 * num_X_checks - num_Z_checks}] rec[{X_check - n - num_X_checks}]"
            )
    else:
        raise NotImplementedError
    no_error_cycle.append("TICK")

    time += gate_1_time + measure_time
    time_cycle += gate_1_time + measure_time

    if rounds > 1:
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, syn_ext))
        time += (rounds - 1) * time_cycle

    # circuit.append(stim.CircuitRepeatBlock(2, no_error_cycle))
    # time += 2 * time_cycle

    """add_idle_noise(
        circuit,
        list(range(n + num_X_checks + num_Z_checks)),
        measure_time,
    )"""
    if basis == "Z":
        circuit.append("M", list(range(n)))
        for Z_check in range(n + num_X_checks, n + num_X_checks + num_Z_checks):
            instr = f"DETECTOR({Z_check}, 1) rec[{Z_check - n - num_X_checks - n - num_Z_checks}]"
            for Z_offset in Z_neighbors:
                if Z_offset != "idle":
                    data = Z_check_Z_offset_2_data[(Z_check, Z_offset)]
                    instr = instr + f" rec[{data - n}]"
            circuit.append_from_stim_program_text(instr)
        for k, z_op in enumerate(Z_logicals):
            z_op_np = np.array(z_op, dtype=np.int8)
            assert z_op_np.shape[0] == n, f"op has shape {z_op_np.shape}"
            support = np.flatnonzero(z_op_np)
            circuit.append_from_stim_program_text(
                f"OBSERVABLE_INCLUDE({k})"
                + "".join([f" rec[{q - n}]" for q in support])
            )
    elif basis == "X":
        circuit.append("MX", list(range(n)))
        for X_check in range(n, n + num_X_checks):
            instr = f"DETECTOR({X_check}, 1) rec[{X_check - n - n - num_X_checks}]"
            for X_offset in X_neighbors:
                if X_offset != "idle":
                    data = X_check_X_offset_2_data[(X_check, X_offset)]
                    instr = instr + f" rec[{data - n}]"
            circuit.append_from_stim_program_text(instr)
        for k, x_op in enumerate(X_logicals):
            x_op_np = np.array(x_op, dtype=np.int8)
            assert x_op_np.shape[0] == n, f"op has shape {x_op_np.shape}"
            support = np.flatnonzero(x_op_np)
            circuit.append_from_stim_program_text(
                f"OBSERVABLE_INCLUDE({k})"
                + "".join([f" rec[{q - n}]" for q in support])
            )
    else:
        return NotImplementedError
    time += measure_time

    return circuit, time
