import stim
import numpy as np
import random


def get_surface_code_circuit_memory(
    d,
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
    circuit = stim.Circuit()
    syn_ext = stim.Circuit()

    # (2d - 1) x (2d - 1) grid

    time = 0.0
    time_cycle = 0.0

    def add_idle_noise(circuit, targets, tick_length):
        if tick_length > 0:
            circuit.append(
                "DEPOLARIZE1",
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

    pos_to_qubit = lambda i, j: i * (2 * d - 1) + j
    data_qubit_to_pos = {}
    X_qubit_to_pos = {}
    Z_qubit_to_pos = {}

    for i in range(2 * d - 1):
        for j in range(2 * d - 1):
            qubit_id = i * (2 * d - 1) + j
            if (i + j) % 2 == 0:
                # data qubit
                circuit.append_from_stim_program_text(
                    f"QUBIT_COORDS({i}, {j}) {qubit_id}"
                )
                data_qubit_to_pos[qubit_id] = (i, j)
            elif i % 2 == 1:
                # X syndrome
                circuit.append_from_stim_program_text(
                    f"QUBIT_COORDS({i}, {j}) {qubit_id}"
                )
                X_qubit_to_pos[qubit_id] = (i, j)
            else:
                # Z syndrome
                circuit.append_from_stim_program_text(
                    f"QUBIT_COORDS({i}, {j}) {qubit_id}"
                )
                Z_qubit_to_pos[qubit_id] = (i, j)

    rec_data_lookup = {
        q: i - len(data_qubit_to_pos) for i, q in enumerate(data_qubit_to_pos.keys())
    }
    rec_X_lookup = {q: i - len(X_qubit_to_pos) for i, q in enumerate(X_qubit_to_pos)}
    rec_Z_lookup = {q: i - len(Z_qubit_to_pos) for i, q in enumerate(Z_qubit_to_pos)}

    if basis == "Z":
        circuit.append(
            "R",
            list(data_qubit_to_pos.keys())
            + list(X_qubit_to_pos.keys())
            + list(Z_qubit_to_pos.keys()),
        )
    elif basis == "X":
        circuit.append("RX", list(data_qubit_to_pos.keys()))
        circuit.append("R", list(X_qubit_to_pos.keys()) + list(Z_qubit_to_pos.keys()))
    add_idle_noise(
        circuit,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        0.0,
    )
    circuit.append("TICK")
    circuit.append("H", list(X_qubit_to_pos.keys()))
    add_idle_noise(
        circuit,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        gate_1_time,
    )
    circuit.append("TICK")

    syn_ext.append("H", list(X_qubit_to_pos.keys()))
    add_idle_noise(
        syn_ext,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        gate_1_time,
    )
    syn_ext.append("TICK")

    time += gate_1_time
    time_cycle += gate_1_time

    delta_X = [(-1, 0), (0, -1), (0, +1), (+1, 0)]
    delta_Z = [(-1, 0), (0, -1), (0, +1), (+1, 0)]
    targets_CX = [{} for _ in range(4)]
    for t in range(4):
        for X_anc in X_qubit_to_pos.keys():
            i, j = X_qubit_to_pos[X_anc]
            if 0 <= j + delta_X[t][1] <= 2 * d - 2:
                assert (
                    data := pos_to_qubit(i + delta_X[t][0], j + delta_X[t][1])
                ) in data_qubit_to_pos
                targets_CX[t][X_anc] = (data, True)
        for Z_anc in Z_qubit_to_pos.keys():
            i, j = Z_qubit_to_pos[Z_anc]
            if 0 <= i + delta_Z[t][0] <= 2 * d - 2:
                assert (
                    data := pos_to_qubit(i + delta_Z[t][0], j + delta_Z[t][1])
                ) in data_qubit_to_pos
                targets_CX[t][Z_anc] = (data, False)

    def generate_cnot_scheduling(cnot_scheduling):
        targets_CX_tick = []
        if cnot_scheduling == "full":
            for t in range(4):
                targets_CX_tick.append([])
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        targets_CX_tick[t].extend([anc, data])
                    else:
                        targets_CX_tick[t].extend([data, anc])
        elif cnot_scheduling == "half":
            for t in range(4):
                targets_CX_tick.extend([[], []])
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        targets_CX_tick[t * 2].extend([anc, data])
                    else:
                        targets_CX_tick[t * 2 + 1].extend([data, anc])
        elif cnot_scheduling == "sqrt":
            for t in range(4):
                targets_CX_tick.extend([[] for _ in range(d)])
                for anc in targets_CX[t]:
                    (data, is_X) = targets_CX[t][anc]
                    if is_X:
                        i, j = X_qubit_to_pos[anc]
                        targets_CX_tick[t * d + ((i - j) // 2) % d].extend([anc, data])
                    else:
                        i, j = Z_qubit_to_pos[anc]
                        targets_CX_tick[t * d + ((i - j) // 2) % d].extend([data, anc])
        elif cnot_scheduling == "minimal":
            for t in range(4):
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
            for t in range(4):
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
    for cnot_targets in targets_CX_tick:
        add_cnot(
            circuit,
            list(data_qubit_to_pos.keys())
            + list(X_qubit_to_pos.keys())
            + list(Z_qubit_to_pos.keys()),
            cnot_targets,
        )
        add_cnot(
            syn_ext,
            list(data_qubit_to_pos.keys())
            + list(X_qubit_to_pos.keys())
            + list(Z_qubit_to_pos.keys()),
            cnot_targets,
        )
        time += gate_2_time
        time_cycle += gate_2_time
        add_idle_noise(
            circuit,
            list(data_qubit_to_pos.keys())
            + list(X_qubit_to_pos.keys())
            + list(Z_qubit_to_pos.keys()),
            pad_time_per_cycle / len(targets_CX_tick),
        )
        add_idle_noise(
            syn_ext,
            list(data_qubit_to_pos.keys())
            + list(X_qubit_to_pos.keys())
            + list(Z_qubit_to_pos.keys()),
            pad_time_per_cycle / len(targets_CX_tick),
        )
        time += pad_time_per_cycle / len(targets_CX_tick)
        time_cycle += pad_time_per_cycle / len(targets_CX_tick)

    circuit.append("H", list(X_qubit_to_pos.keys()))
    add_idle_noise(
        circuit,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        gate_1_time,
    )
    circuit.append("TICK")
    add_idle_noise(
        circuit,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        measure_time,
    )
    if basis == "Z":
        circuit.append("MR", list(X_qubit_to_pos.keys()) + list(Z_qubit_to_pos.keys()))
        for Z_anc in Z_qubit_to_pos.keys():
            circuit.append_from_stim_program_text(
                f"DETECTOR({Z_qubit_to_pos[Z_anc][0]}, {Z_qubit_to_pos[Z_anc][1]}, 0) rec[{rec_Z_lookup[Z_anc]}]"
            )
    elif basis == "X":
        circuit.append("MR", list(Z_qubit_to_pos.keys()) + list(X_qubit_to_pos.keys()))
        for X_anc in X_qubit_to_pos.keys():
            circuit.append_from_stim_program_text(
                f"DETECTOR({X_qubit_to_pos[X_anc][0]}, {X_qubit_to_pos[X_anc][1]}, 0) rec[{rec_X_lookup[X_anc]}]"
            )
    else:
        raise NotImplementedError
    circuit.append("TICK")

    syn_ext.append("H", list(X_qubit_to_pos.keys()))
    add_idle_noise(
        syn_ext,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        gate_1_time,
    )
    syn_ext.append("TICK")
    add_idle_noise(
        syn_ext,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        measure_time,
    )
    if basis == "Z":
        syn_ext.append("MR", list(X_qubit_to_pos.keys()) + list(Z_qubit_to_pos.keys()))
        syn_ext.append_from_stim_program_text("SHIFT_COORDS(0, 0, 1)")
        for X_anc in X_qubit_to_pos.keys():
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({X_qubit_to_pos[X_anc][0]}, {X_qubit_to_pos[X_anc][1]}, 0) rec[{rec_X_lookup[X_anc] - 2 * len(Z_qubit_to_pos) - len(X_qubit_to_pos)}] rec[{rec_X_lookup[X_anc] - len(Z_qubit_to_pos)}]"
            )
        for Z_anc in Z_qubit_to_pos.keys():
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({Z_qubit_to_pos[Z_anc][0]}, {Z_qubit_to_pos[Z_anc][1]}, 0) rec[{rec_Z_lookup[Z_anc] - len(Z_qubit_to_pos) - len(X_qubit_to_pos)}] rec[{rec_Z_lookup[Z_anc]}]"
            )
    elif basis == "X":
        syn_ext.append("MR", list(Z_qubit_to_pos.keys()) + list(X_qubit_to_pos.keys()))
        syn_ext.append_from_stim_program_text("SHIFT_COORDS(0, 0, 1)")
        for Z_anc in Z_qubit_to_pos.keys():
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({Z_qubit_to_pos[Z_anc][0]}, {Z_qubit_to_pos[Z_anc][1]}, 0) rec[{rec_Z_lookup[Z_anc] - len(Z_qubit_to_pos) - 2 * len(X_qubit_to_pos)}] rec[{rec_Z_lookup[Z_anc] - len(X_qubit_to_pos)}]"
            )
        for X_anc in X_qubit_to_pos.keys():
            syn_ext.append_from_stim_program_text(
                f"DETECTOR({X_qubit_to_pos[X_anc][0]}, {X_qubit_to_pos[X_anc][1]}, 0) rec[{rec_X_lookup[X_anc] - len(Z_qubit_to_pos) - len(X_qubit_to_pos)}] rec[{rec_X_lookup[X_anc]}]"
            )
    else:
        raise NotImplementedError
    syn_ext.append("TICK")

    time += gate_1_time + measure_time
    time_cycle += gate_1_time + measure_time

    if rounds > 1:
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, syn_ext))
        time += (rounds - 1) * time_cycle

    add_idle_noise(
        circuit,
        list(data_qubit_to_pos.keys())
        + list(X_qubit_to_pos.keys())
        + list(Z_qubit_to_pos.keys()),
        measure_time,
    )
    if basis == "Z":
        circuit.append("M", list(data_qubit_to_pos.keys()))
        for Z_anc in Z_qubit_to_pos:
            instr = f"DETECTOR({Z_qubit_to_pos[Z_anc][0]}, {Z_qubit_to_pos[Z_anc][1]}, 1) rec[{rec_Z_lookup[Z_anc] - len(data_qubit_to_pos)}]"
            for t in range(4):
                i, j = Z_qubit_to_pos[Z_anc]
                if 0 <= i + delta_Z[t][0] <= 2 * d - 2:
                    assert (
                        data := pos_to_qubit(i + delta_Z[t][0], j + delta_Z[t][1])
                    ) in data_qubit_to_pos
                    instr = instr + f" rec[{rec_data_lookup[data]}]"
            circuit.append_from_stim_program_text(instr)
        circuit.append_from_stim_program_text(
            "OBSERVABLE_INCLUDE(0)"
            + "".join(
                [
                    f" rec[{rec_data_lookup[pos_to_qubit(j, 0)]}]"
                    for j in range(0, 2 * d - 1, 2)
                ]
            )
        )
    elif basis == "X":
        circuit.append("MX", list(data_qubit_to_pos.keys()))
        for X_anc in X_qubit_to_pos:
            instr = f"DETECTOR({X_qubit_to_pos[X_anc][0]}, {X_qubit_to_pos[X_anc][1]}, 1) rec[{rec_X_lookup[X_anc] - len(data_qubit_to_pos)}]"
            for t in range(4):
                i, j = X_qubit_to_pos[X_anc]
                if 0 <= j + delta_X[t][1] <= 2 * d - 2:
                    assert (
                        data := pos_to_qubit(i + delta_X[t][0], j + delta_X[t][1])
                    ) in data_qubit_to_pos
                    instr = instr + f" rec[{rec_data_lookup[data]}]"
            circuit.append_from_stim_program_text(instr)
        circuit.append_from_stim_program_text(
            "OBSERVABLE_INCLUDE(0)"
            + "".join(
                [
                    f" rec[{rec_data_lookup[pos_to_qubit(0, i)]}]"
                    for i in range(0, 2 * d - 1, 2)
                ]
            )
        )
    else:
        return NotImplementedError
    time += measure_time

    return circuit, time
