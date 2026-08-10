import hashlib
import json
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_service_names(service_file, cluster_file):
    with open(service_file, "r", encoding="utf-8") as file:
        services = json.load(file)["service_names"]
    with open(cluster_file, "r", encoding="utf-8") as file:
        cluster_configs = json.load(file)
    cluster_names = list(cluster_configs.keys())
    return [
        service_name.replace("-service", "") + "_" + cluster_name
        for service_name in services
        for cluster_name in cluster_names
    ]


ADVANCED_TICKET_SERVICE_WEIGHTS = {
    "ts-auth": 1.00,
    "ts-route": 0.95,
    "ts-route-plan": 0.90,
    "ts-travel-plan": 0.90,
    "ts-travel": 0.85,
    "ts-travel2": 0.75,
    "ts-station": 0.70,
    "ts-train": 0.70,
    "ts-ticketinfo": 0.65,
    "ts-price": 0.55,
    "ts-basic": 0.45,
    "ts-auth-mongo": 0.25,
    "ts-route-mongo": 0.25,
    "ts-travel-mongo": 0.20,
    "ts-travel2-mongo": 0.20,
    "ts-station-mongo": 0.18,
    "ts-train-mongo": 0.18,
    "ts-price-mongo": 0.15,
}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _stable_service_jitter(service_name):
    digest = hashlib.sha256(service_name.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / float(2**32 - 1)
    return 0.90 + 0.20 * value


def _service_base_name(service_name):
    return service_name.rsplit("_", 1)[0]


def _cluster_share(service_name):
    if service_name.endswith("_cluster2"):
        return 3.0 / 8.0
    if service_name.endswith("_cluster3"):
        return 5.0 / 8.0
    return 0.5


def _service_weight(service_name):
    base_name = _service_base_name(service_name)
    mode = os.environ.get("HEADLIGHT_BOOTSTRAP_TRAINTICKET_MODE", "advanced-ticket")
    if mode == "advanced-ticket":
        return ADVANCED_TICKET_SERVICE_WEIGHTS.get(base_name, 0.03)
    if "mongo" in base_name:
        return 0.20
    return 0.60


def _workload_users_at(step_index, total_steps):
    start_users = _env_float("HEADLIGHT_BOOTSTRAP_START_USERS", 5)
    end_users = _env_float("HEADLIGHT_BOOTSTRAP_END_USERS", start_users)
    duration = max(1.0, _env_float("HEADLIGHT_BOOTSTRAP_DURATION_SECONDS", total_steps))
    ramp = max(1.0, _env_float("HEADLIGHT_BOOTSTRAP_RAMP_SECONDS", duration))
    hold = max(0.0, _env_float("HEADLIGHT_BOOTSTRAP_HOLD_SECONDS", duration - ramp))
    elapsed = duration * step_index / max(1, total_steps - 1)
    if elapsed <= ramp:
        return start_users + (end_users - start_users) * (elapsed / ramp)
    if elapsed <= ramp + hold:
        return end_users
    remaining = max(1.0, duration - ramp - hold)
    return end_users + (start_users - end_users) * ((elapsed - ramp - hold) / remaining)


def _desired_pods(users, service_name):
    weighted_users = users * _service_weight(service_name) * _cluster_share(service_name)
    users_per_pod = max(0.5, _env_float("HEADLIGHT_BOOTSTRAP_USERS_PER_POD", 2.0))
    max_replicas = 3 if service_name.endswith("_cluster2") else 5
    return float(np.clip(np.ceil(weighted_users / users_per_pod), 1, max_replicas))


def _workload_shaped_history(history_length, input_size, service_name=""):
    if input_size < 4:
        raise ValueError(f"expected at least 4 input features, got {input_size}")
    idle_steps = int(0.5 * history_length)
    profile_steps = max(1, history_length)
    data = np.zeros((idle_steps + profile_steps, input_size), dtype=np.float32)
    col_pod, col_res, col_req, col_cpu = 0, 1, 2, 3

    data[:idle_steps, col_pod] = 1.0
    jitter = _stable_service_jitter(service_name or "default")
    users_per_pod = max(0.5, _env_float("HEADLIGHT_BOOTSTRAP_USERS_PER_POD", 2.0))
    for offset in range(profile_steps):
        row = idle_steps + offset
        users = _workload_users_at(offset, profile_steps)
        pods = _desired_pods(users, service_name)
        weighted_users = users * _service_weight(service_name) * _cluster_share(service_name) * jitter
        saturation = weighted_users / max(1.0, pods * users_per_pod)
        data[row, col_pod] = pods
        data[row, col_res] = np.clip(0.08 + 0.10 * saturation + 0.20 * max(0.0, saturation - 1.0) ** 2, 0.01, 5.0)
        data[row, col_req] = max(0.0, weighted_users / 10.0)
        data[row, col_cpu] = min(0.95, 0.03 + 0.12 * saturation)
    return data


def _synthetic_history(history_length, input_size):
    return _workload_shaped_history(history_length, input_size)


def build_training_arrays(feature_history_folder, service_names, history_length, input_size, step_num_in, step_num_out):
    initial_node_histories = []
    for service_name in service_names:
        history_path = os.path.join(feature_history_folder, f"{service_name}_history.npy")
        if os.path.exists(history_path):
            data = np.load(history_path).astype(np.float32)
        else:
            data = np.array([], dtype=np.float32)

        if data.ndim != 2 or data.shape[0] < history_length:
            data = _workload_shaped_history(history_length, input_size, service_name)
        elif data.shape[1] != input_size:
            raise ValueError(f"{history_path} has {data.shape[1]} features, expected {input_size}")

        scaler = StandardScaler()
        data[:, 1:] = scaler.fit_transform(data[:, 1:]).astype(np.float32)
        initial_node_histories.append(data)

        scaler_path = os.path.join(feature_history_folder, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    effective_history_length = min(h.shape[0] for h in initial_node_histories)
    node_histories = [h[-effective_history_length:] for h in initial_node_histories]

    x_list, y_list = [], []
    for t in range(effective_history_length - step_num_in - step_num_out + 1):
        x = np.zeros((step_num_in, len(service_names), input_size), dtype=np.float32)
        y = np.zeros((step_num_out, len(service_names)), dtype=np.float32)
        for node_id, history in enumerate(node_histories):
            x[:, node_id, :] = history[t : t + step_num_in]
            y[:, node_id] = history[t + step_num_in : t + step_num_in + step_num_out, 0]
        x_list.append(x)
        y_list.append(y)

    if not x_list:
        raise ValueError("not enough feature history to build training samples")
    return np.stack(x_list), np.stack(y_list)


def build_training_dataset(
    feature_history_folder,
    history_length,
    input_size,
    num_nodes,
    step_num_in,
    step_num_out,
    batch_size,
    service_names,
    dataloader_workers=0,
    train_split=1.0,
):
    x, y = build_training_arrays(feature_history_folder, service_names, history_length, input_size, step_num_in, step_num_out)
    if x.shape[1] != num_nodes:
        raise ValueError(f"expected {num_nodes} nodes, found {x.shape[1]}")
    np.save(os.path.join(feature_history_folder, "X.npy"), x)
    np.save(os.path.join(feature_history_folder, "Y.npy"), y)

    split_index = int(len(x) * train_split)
    split_index = max(1, min(split_index, len(x)))
    dataset = TensorDataset(torch.tensor(x[:split_index], dtype=torch.float32), torch.tensor(y[:split_index], dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)


def build_train_validation_loaders(
    feature_history_folder,
    service_names,
    history_length,
    input_size,
    step_num_in,
    step_num_out,
    batch_size,
    dataloader_workers,
    train_split,
):
    x, y = build_training_arrays(feature_history_folder, service_names, history_length, input_size, step_num_in, step_num_out)
    np.save(os.path.join(feature_history_folder, "X.npy"), x)
    np.save(os.path.join(feature_history_folder, "Y.npy"), y)
    split_index = int(len(x) * train_split)
    split_index = max(1, min(split_index, len(x) - 1))

    train_dataset = TensorDataset(torch.tensor(x[:split_index], dtype=torch.float32), torch.tensor(y[:split_index], dtype=torch.float32))
    validation_dataset = TensorDataset(torch.tensor(x[split_index:], dtype=torch.float32), torch.tensor(y[split_index:], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)
    return train_loader, validation_loader, split_index, len(x) - split_index


def main():
    from headlight.training import main as offline_main

    return offline_main()


if __name__ == "__main__":
    raise SystemExit(main())
