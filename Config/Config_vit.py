


def get_config_dict():
    dataset_name = "cifar10"



    dataset = dict(
        name= dataset_name
    )
    args = dict(
        gpu_id='0',
        batch_size=16,
        network_name="vit",
        epochs=100,
        num_workers=3
    )
    solver = dict(
        optimizer="adam",
        scheduler='steplr',
        step_size=5,
        gamma=0.95,
        loss="crossentropy",
        lr=1e-3,
        weight_decay=5e-5,
        print_freq=20,
    )
    model = dict(
        resume='/home/sjpark/PycharmProjects/vehicle_segmentation/checkpoints/',  # weight_file
        mode='train',
        save_dir='./runs/train',
        checkpoint='/storage/sjpark/vehicle_data/checkpoints'  # checkpoint_path
    )
    config = dict(
        args=args,
        dataset=dataset,
        solver=solver,
        model=model
    )

    return config
