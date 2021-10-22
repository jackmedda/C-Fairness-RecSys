import torch
import numpy as np
from torch.nn.functional import softplus
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from optimizer import train_model

inv_softplus = lambda x: x + torch.log(-torch.expm1(-x))


class LbmOrdinal(torch.nn.Module):
    def __init__(self, *, temperature=0.6, device=None):
        super(LbmOrdinal, self).__init__()
        self.temperature = temperature
        self.params = torch.nn.ParameterDict({})
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cpu")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    @property
    def alpha_1(self):
        return (
            torch.nn.functional.softmax(self.params["r_alpha_1"], 1)
            .double()
            .reshape(self.nq, 1)
        )

    @property
    def alpha_2(self):
        return (
            torch.nn.functional.softmax(self.params["r_alpha_2"], 1)
            .double()
            .reshape(self.nl, 1)
        )

    @property
    def tau_1(self):
        return torch.nn.functional.softmax(self.params["r_tau_1"], 1).double()
        return expand_simplex(
            torch.sigmoid(self.params["r_tau_1"]).reshape(
                self.n1, self.nq - 1
            ),
            self.device,
        ).reshape(self.n1, self.nq)

    @property
    def tau_2(self):
        return torch.nn.functional.softmax(self.params["r_tau_2"], 1).double()
        return expand_simplex(
            torch.sigmoid(self.params["r_tau_2"]).reshape(
                self.n2, self.nl - 1
            ),
            self.device,
        ).reshape(self.n2, self.nl)

    @property
    def beta_mar(self):
        return self.params["r_beta_mar"].double()

    @property
    def pi(self):
        return self.params["r_pi"].double()

    @property
    def sigma(self):
        return softplus(self.params["r_sigma"])

    @property
    def sigma_sq_row(self):
        return softplus(self.params["r_sigma_sq_row"])

    @property
    def sigma_sq_col(self):
        return softplus(self.params["r_sigma_sq_col"])

    @property
    def sigma_sq_row_mean(self):
        return softplus(self.params["r_sigma_sq_row_mean"])

    @property
    def sigma_sq_col_mean(self):
        return softplus(self.params["r_sigma_sq_col_mean"])

    @property
    def sigma_sq_beta_x(self):
        return softplus(self.params["r_sigma_sq_beta_x"])

    @property
    def mu(self):
        return self.params["r_mu"].double()

    @property
    def nu_row(self):
        return self.params["r_nu_row"]

    @property
    def nu_col(self):
        return self.params["r_nu_col"]

    @property
    def eta_row(self):
        return self.params["r_eta_row"]

    @property
    def eta_col(self):
        return self.params["r_eta_col"]

    @property
    def rho_row(self):
        return softplus(self.params["r_rho_row"])

    @property
    def rho_col(self):
        return softplus(self.params["r_rho_col"])

    @property
    def psi_row(self):
        return softplus(self.params["r_psi_row"])

    @property
    def psi_col(self):
        return softplus(self.params["r_psi_col"])

    # Here beta_x is a latent variable
    @property
    def beta_x(self):
        return self.params["r_beta_x"].double()

    @property
    def rho_beta_x(self):
        return softplus(self.params["r_rho_beta_x"])

    def fit(self, X, nq, nl, cov=None, init_parameters=None, **kwargs):
        # X : the sparse rating table
        # nq: number of row groups
        # nl: number of columns groups
        # cov : protected attributes

        x_shape = X.shape
        self.n1, self.n2 = X.shape[-2:]
        self.nq, self.nl = nq, nl
        self.nb_categories = int(X.max().item())

        sparsity = (
            (X == 0).float().mean().item()
            if isinstance(X, torch.Tensor)
            else (X == 0).mean().item()
        )
        nb_covariates = cov.shape[-1] if cov is not None else 0

        self.params.update(
            random_init_lbm_ordinal_mar(
                self.n1,
                self.n2,
                self.nq,
                self.nl,
                sparsity,
                nb_covariates=nb_covariates,
                device=self.device,
            )
        )

        # Fixed levels for ordinal regression.
        self._theta = (
            torch.arange(0, self.nb_categories - 1, device=self.device) + 0.5
        )
        self.register_buffer("theta", self._theta)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device)
        else:
            X = X.to(self.device)

        if cov is not None:
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, device=self.device)
            else:
                cov = cov.to(self.device)

        train_model(self, X, covariates=cov, **kwargs)
        return self

    def decode(self, rows_sampled, columns_sampled, cov=None):
        eps = 1e-8
        gauss_row_sampled = sample_gaussian(
            self.nu_row[rows_sampled], self.rho_row[rows_sampled,]
        )
        gauss_col_sampled = sample_gaussian(
            self.nu_col[:, columns_sampled], self.rho_col[:, columns_sampled]
        )
        p = (
            self.mu
            + gauss_row_sampled.reshape(rows_sampled.flatten().shape[0], 1)
            + gauss_col_sampled.reshape(1, columns_sampled.flatten().shape[0])
        )
        prob_observed = torch.sigmoid(p)
        prob_observed = torch.clamp(prob_observed, min=eps, max=1 - eps)

        Y1_sampled = sample_categorial(
            self.temperature, self.tau_1[rows_sampled,]
        )
        Y2_sampled = sample_categorial(
            self.temperature, self.tau_2[columns_sampled,]
        )
        mean_row_sampled = sample_gaussian(
            self.eta_row[rows_sampled], self.psi_row[rows_sampled,]
        )
        mean_col_sampled = sample_gaussian(
            self.eta_col[:, columns_sampled], self.psi_col[:, columns_sampled]
        )

        p = (
            Y1_sampled @ self.pi @ Y2_sampled.T
            + mean_row_sampled
            + mean_col_sampled
            # + cov.squeeze() * beta_x
        )

        if cov is not None:
            beta_x = sample_gaussian(
                self.beta_x[:, columns_sampled],
                self.rho_beta_x[:, columns_sampled],
            )
            p += cov.squeeze() * beta_x

        m = torch.distributions.Normal(p, self.sigma)
        cumu = m.cdf(self.theta.reshape(-1, 1, 1))
        prob_X = torch.cat((cumu[:1], cumu[1:] - cumu[:-1], 1 - cumu[-1:]))

        # For stability reasons.
        prob_X = torch.clamp(prob_X, min=eps, max=1 - eps)
        prob_X /= prob_X.sum(0).unsqueeze(0)

        # Probabilities of observations in classes.
        # The order Non-obseved, ordered values of the vector of
        # probabilities must match with the (x+1) classes.
        prob = prob_X * prob_observed.unsqueeze(0)
        prob = torch.cat(((1 - prob.sum(0)).unsqueeze(0), prob))

        prob = torch.clamp(prob, min=eps, max=1 - eps)
        prob /= prob.sum(0).unsqueeze(0)

        return prob

    def batch_entropy(self, rows_sampled, columns_sampled, cov=None):
        nb_samples_row = rows_sampled.size()[0]
        nb_samples_col = columns_sampled.size()[0]
        batch_ratio_rows = columns_sampled.size()[0] / self.n2
        batch_ratio_cols = rows_sampled.size()[0] / self.n1

        entropy = (
            batch_entropy_categorical(
                batch_ratio_rows, self.tau_1[rows_sampled]
            )
            + batch_entropy_categorical(
                batch_ratio_cols, self.tau_2[columns_sampled]
            )
            + batch_entropy_gaussian(
                batch_ratio_rows, nb_samples_row, self.rho_row[rows_sampled]
            )
            + batch_entropy_gaussian(
                batch_ratio_cols,
                nb_samples_col,
                self.rho_col[:, columns_sampled],
            )
            + batch_entropy_gaussian(
                batch_ratio_rows, nb_samples_row, self.psi_row[rows_sampled]
            )
            + batch_entropy_gaussian(
                batch_ratio_cols,
                nb_samples_col,
                self.psi_col[:, columns_sampled],
            )
        )
        if cov is not None:
            entropy += batch_entropy_gaussian(
                batch_ratio_cols,
                nb_samples_col,
                self.rho_beta_x[:, columns_sampled],
            )
        return entropy

    def batch_expectation_loglike_latent(
        self, rows_sampled, columns_sampled, cov=None
    ):
        nb_samples_row = rows_sampled.size()[0]
        nb_samples_col = columns_sampled.size()[0]
        batch_ratio_rows = columns_sampled.size()[0] / self.n2
        batch_ratio_cols = rows_sampled.size()[0] / self.n1
        expectation = (
            batch_categorial_expectation_loglike(
                batch_ratio_rows, self.tau_1[rows_sampled], self.alpha_1
            )
            + batch_categorial_expectation_loglike(
                batch_ratio_cols, self.tau_2[columns_sampled], self.alpha_2
            )
            + batch_gaussian_expectation_loglike(
                batch_ratio_rows,
                nb_samples_row,
                self.sigma_sq_row,
                self.nu_row[rows_sampled],
                self.rho_row[rows_sampled],
            )
            + batch_gaussian_expectation_loglike(
                batch_ratio_cols,
                nb_samples_col,
                self.sigma_sq_col,
                self.nu_col[:, columns_sampled],
                self.rho_col[:, columns_sampled],
            )
            + batch_gaussian_expectation_loglike(
                batch_ratio_rows,
                nb_samples_row,
                self.sigma_sq_row_mean,
                self.eta_row[rows_sampled],
                self.psi_row[rows_sampled],
            )
            + batch_gaussian_expectation_loglike(
                batch_ratio_cols,
                nb_samples_col,
                self.sigma_sq_col_mean,
                self.eta_col[:, columns_sampled],
                self.psi_col[:, columns_sampled],
            )
        )
        if cov is not None:
            expectation += batch_gaussian_expectation_loglike(
                batch_ratio_cols,
                nb_samples_col,
                self.sigma_sq_beta_x,
                self.beta_x[:, columns_sampled],
                self.rho_beta_x[:, columns_sampled],
            )
        return expectation

    def forward(self, x, rows_sampled, columns_sampled, cov=None):
        reconstructed_x = self.decode(rows_sampled, columns_sampled, cov)

        criterion = (
            -self.batch_entropy(rows_sampled, columns_sampled, cov)
            - self.batch_expectation_loglike_latent(
                rows_sampled, columns_sampled, cov
            )
            + (
                torch.nn.functional.nll_loss(
                    torch.log(reconstructed_x).flatten(1, 2).T,
                    x.flatten(),
                    reduction="sum",
                )
            ).sum()
        )
        return criterion

    def get_ll(self, X, nq, nl, cov=None):
        x_shape = X.shape
        self.n1, self.n2 = X.shape[-2:]
        self.nq, self.nl = nq, nl
        self.nb_categories = int(X.max().item())
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device)
        else:
            X = X.to(self.device)

        if cov is not None:
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, device=self.device)
            else:
                cov = cov.to(self.device)

        # Fixed levels for ordinal regression.
        self._theta = (
            torch.arange(0, self.nb_categories - 1, device=self.device) + 0.5
        )
        self.register_buffer("theta", self._theta)
        with torch.no_grad():
            rows_sampled = torch.arange(X.shape[0], device=self.device)
            columns_sampled = torch.arange(X.shape[1], device=self.device)
            reconstructed_x = self.decode(rows_sampled, columns_sampled, cov)
            nll = (
                -self.batch_entropy(rows_sampled, columns_sampled, cov)
                - self.batch_expectation_loglike_latent(
                    rows_sampled, columns_sampled, cov
                )
                + (
                    torch.nn.functional.nll_loss(
                        torch.log(reconstructed_x).flatten(1, 2).T,
                        X.flatten(),
                        reduction="sum",
                    )
                ).sum()
            )
        return nll

    def to_numpy(self):
        with torch.no_grad():
            params = {
                k[2:]: getattr(self, k[2:]).cpu().numpy()
                for k in self.params.keys()
            }
        return params


def random_init_lbm_ordinal_mar(
    n1, n2, nq, nl, sparsity, *, nb_covariates=0, device=torch.device("cpu")
):
    r_alpha_1 = torch.ones((1, nq), device=device)
    r_alpha_2 = torch.ones((1, nl), device=device)
    r_tau_1 = (
        torch.distributions.Multinomial(
            probs=torch.ones(nq, device=device) / nq
        ).sample((n1,))
        * 4
    )
    r_tau_2 = (
        torch.distributions.Multinomial(
            probs=torch.ones(nl, device=device) / nl
        ).sample((n2,))
        * 4
    )
    r_pi = torch.zeros((nq, nl), device=device).float()
    r_sigma = inv_softplus(0.8 * torch.ones(1, device=device).double())
    r_mu = torch.logit(sparsity * torch.ones(1, device=device))
    r_nu_row = torch.zeros((n1, 1), device=device)
    r_nu_col = torch.zeros((1, n2), device=device)
    r_rho_row = inv_softplus(torch.ones((n1, 1), device=device).double())
    r_rho_col = inv_softplus(torch.ones((1, n2), device=device).double())
    r_sigma_sq_row = inv_softplus(torch.ones(1, device=device).double())
    r_sigma_sq_col = inv_softplus(torch.ones(1, device=device).double())
    r_eta_row = torch.zeros((n1, 1), device=device)
    r_eta_col = torch.zeros((1, n2), device=device)
    r_psi_row = inv_softplus(torch.ones((n1, 1), device=device).double())
    r_psi_col = inv_softplus(torch.ones((1, n2), device=device).double())
    r_sigma_sq_row_mean = inv_softplus(torch.ones(1, device=device).double())
    r_sigma_sq_col_mean = inv_softplus(torch.ones(1, device=device).double())

    res = {
        "r_tau_1": torch.nn.Parameter(r_tau_1),
        "r_tau_2": torch.nn.Parameter(r_tau_2),
        "r_alpha_1": torch.nn.Parameter(r_alpha_1),
        "r_alpha_2": torch.nn.Parameter(r_alpha_2),
        "r_eta_row": torch.nn.Parameter(r_eta_row),
        "r_eta_col": torch.nn.Parameter(r_eta_col),
        "r_psi_row": torch.nn.Parameter(r_psi_row),
        "r_psi_col": torch.nn.Parameter(r_psi_col),
        "r_sigma_sq_row_mean": torch.nn.Parameter(r_sigma_sq_row_mean),
        "r_sigma_sq_col_mean": torch.nn.Parameter(r_sigma_sq_col_mean),
        "r_pi": torch.nn.Parameter(r_pi),
        "r_mu": torch.nn.Parameter(r_mu),
        "r_sigma": torch.nn.Parameter(r_sigma),
        "r_nu_row": torch.nn.Parameter(r_nu_row),
        "r_nu_col": torch.nn.Parameter(r_nu_col),
        "r_rho_row": torch.nn.Parameter(r_rho_row),
        "r_rho_col": torch.nn.Parameter(r_rho_col),
        "r_sigma_sq_row": torch.nn.Parameter(r_sigma_sq_row),
        "r_sigma_sq_col": torch.nn.Parameter(r_sigma_sq_col),
    }

    if nb_covariates:
        r_beta_x = torch.zeros((1, n2), device=device)
        r_rho_beta_x = inv_softplus(
            torch.ones((1, n2), device=device).double()
        )
        r_sigma_sq_beta_x = inv_softplus(torch.ones(1, device=device).double())
        res.update(
            {
                "r_beta_x": torch.nn.Parameter(r_beta_x),
                "r_rho_beta_x": torch.nn.Parameter(r_rho_beta_x),
                "r_sigma_sq_beta_x": torch.nn.Parameter(r_sigma_sq_beta_x),
            }
        )

    return res


def sample_categorial(temperature, prob):
    return RelaxedOneHotCategorical(temperature, probs=prob).rsample().double()


def sample_gaussian(mu, var):
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return mu + eps * std


def batch_entropy_categorical(batch_ratio, tau):
    return batch_ratio * (-torch.sum(tau * torch.log(tau)))


def batch_categorial_expectation_loglike(batch_ratio, tau, alpha):
    return batch_ratio * (tau.sum(0) @ torch.log(alpha))


def batch_entropy_gaussian(batch_ratio, nb_samples, rho):
    device = rho.device
    return (
        0.5
        * batch_ratio
        * (
            nb_samples
            * (
                torch.log(
                    torch.tensor(2 * np.pi, dtype=torch.double, device=device)
                )
                + 1
            )
            + torch.sum(torch.log(rho))
        )
    )


def batch_gaussian_expectation_loglike(
    batch_ratio, nb_samples, sigma_sq, nu, rho
):
    device = nu.device
    return batch_ratio * (
        -nb_samples
        / 2
        * (
            torch.log(
                torch.tensor(2 * np.pi, dtype=torch.double, device=device)
            )
            + torch.log(sigma_sq)
        )
        - 1 / (2 * sigma_sq) * torch.sum(rho + nu ** 2)
    )
