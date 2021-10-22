from scipy.signal import savgol_filter
import torch
import numpy as np
import itertools
import collections


def train_model(
    model,
    X,
    *,
    covariates=None,
    lr=1e-2,
    max_epoch=2000,
    batch_size=(100, 120),
    optimizer=None,
    optimizer_options={},
    scheduler_options={},
    callback=None,
    regu=0,
):
    default_optimizer_options = {"lr": lr}
    default_scheduler_options = {
        "mode": "min",
        "factor": 0.2,
        "patience": 10,
        "cooldown": 10,
    }
    default_optimizer_options.update(optimizer_options)
    default_scheduler_options.update(scheduler_options)

    optimizer_class = (
        optimizer
        if (optimizer and issubclass(torch.optim.Adam, torch.optim.Optimizer))
        else torch.optim.Adam
    )

    optimizer = optimizer_class(
        model.parameters(), **default_optimizer_options
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **default_scheduler_options
    )

    multi_dim = True if len(X.shape) == 3 else False

    if covariates is not None:
        assert X.shape[-2:] == covariates.shape[:2], "Mistaken dimensions"
    else:
        cov_batch = None

    loss_history = []
    for epoch in range(0, max_epoch):
        rows = torch.randperm(model.n1)
        columns = torch.randperm(model.n2)
        a, b = rows.split(batch_size[0]), columns.split(batch_size[1])
        train_loss = 0

        if epoch % 50 == 0:
            optimizer.state = collections.defaultdict(dict)
        for i_batch, (rows_sampled, columns_sampled) in enumerate(
            sorted(itertools.product(a, b), key=lambda x: np.random.uniform())
        ):

            X_batch = (
                X[:, rows_sampled][:, :, columns_sampled]
                if multi_dim
                else X[rows_sampled][:, columns_sampled]
            )

            if covariates is not None:
                cov_batch = covariates[rows_sampled][:, columns_sampled]
            optimizer.zero_grad()

            loss = model(X_batch, rows_sampled, columns_sampled, cov_batch)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            print(
                "Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(
                    epoch,
                    i_batch + 1,
                    len(a) * len(b),
                    loss.item()
                    / (rows_sampled.size()[0] * columns_sampled.size()[0]),
                ),
                end="\r",
            )
        loss_history.append(train_loss)
        savgol_window = 21
        savgol_polynom = 3
        if epoch > 1:
            if epoch > savgol_window:
                yhat = savgol_filter(
                    loss_history, savgol_window, savgol_polynom
                )
        if epoch > 50:
            scheduler.step(yhat[-1])
        actual_rtol = 0
        if epoch > 25:
            actual_rtol = (yhat[-20] - yhat[-1]) / yhat[-20]
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print("Converged.")
                # plt.close()
                break

        print(
            f"""Epoch: {epoch:4} \
            Loss: {train_loss:.3E} \
            Temp: {model.temperature:.3f} \
            Lr: {optimizer.param_groups[0]['lr']:.1E} \
            """
        )
        if callback:
            callback(model, epoch)
