from .label_replacement import apply_class_label_replacement


def poison_data(distributed_dataset, num_workers, poisoned_worker_ids, replacement_method):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    # TODO: Add support for multiple replacement methods?
    poisoned_dataset = []

    class_labels = list(set(distributed_dataset[0][1]))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            poisoned_dataset.append(
                apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1],
                                              replacement_method))
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    return poisoned_dataset
