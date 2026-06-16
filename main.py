from dataclasses import dataclass
import numpy as np

from data import prep_dataset, global_split
from eval import lr_effect, lambda_effect, evaluate, cross_validation, cost_each_epoch, size_effect_on_accuracy, size_effect_on_cost 
from build_network import build_model
from ucimlrepo import fetch_ucirepo 


@dataclass
class Config:
    fold_num: int = 3
    num_epochs: int = 50
    final_epochs: int = 150
    learning_rates: tuple = (0.005, 0.01, 0.02, 0.05, 0.1, 0.5)
    lambdas: tuple = (0,0.002,0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24)
    sizes: range = range(50,600,50)
    seed: int = 42


class DatasetBuilder:
    @staticmethod
    def build(X_train, y_train, X_test, y_test, size, add_feat):
        return prep_dataset(
            X_train,
            y_train,
            X_test,
            y_test,
            size,
            add_feat=add_feat
        )


class Trainer:
    def __init__(self, config):
        self.cfg = config

    def best_lr(self, dataset, factory, lambda_):
        return lr_effect(
            dataset,
            self.cfg.learning_rates,
            self.cfg.fold_num,
            self.cfg.num_epochs,
            factory,
            lambda_,
            plot=False,
            seed=self.cfg.seed
        )

    def best_lambda(self, dataset, factory, lr):
        best_cost = float("inf")
        best_lambda = self.cfg.lambdas[0]

        for lambda_ in self.cfg.lambdas:
            _, cost = cross_validation(
                dataset.X_train,
                dataset.Y_train,
                self.cfg.fold_num,
                self.cfg.num_epochs,
                factory,
                lr,
                lambda_,
                seed=self.cfg.seed
            )

            if cost < best_cost:
                best_cost = cost
                best_lambda = lambda_

        return best_lambda

    def fit_model(self, dataset, factory, lr, lambda_):
        net = factory()
        net.fit(
            dataset.X_train,
            dataset.Y_train,
            self.cfg.final_epochs,
            lr,
            lambda_
        )
        return net

    def train_best(self, dataset, factory, lr=None, lambda_=None):
        if lr is None and lambda_ is None:
            lr = self.best_lr(dataset, factory, 0.0001)
            lambda_ = self.best_lambda(dataset, factory, lr)
        elif lr is None:
            lr = self.best_lr(dataset, factory, lambda_)
        elif lambda_ is None:
            lambda_ = self.best_lambda(dataset, factory, lr)

        net = self.fit_model(dataset, factory, lr, lambda_)

        _, acc, pred = evaluate(
            net,
            dataset.X_test,
            dataset.Y_test
        )

        return {
            "net": net,
            "acc": acc,
            "pred": pred,
            "lr": lr,
            "lambda": lambda_
        }


class Experiment:
    def __init__(self, trainer, config):
        self.trainer = trainer
        self.cfg = config

    def run_single(self, x_train, y_train, x_test, y_test, add_feat, lr=None, lambda_=None):
        dataset = DatasetBuilder.build(
            x_train,
            y_train,
            x_test,
            y_test,
            len(x_train),
            add_feat
        )

        input_dim = dataset.X_train.shape[1]

        def factory():
            return build_model(input_dim=input_dim, seed=self.cfg.seed)

        result = self.trainer.train_best(dataset, factory, lr, lambda_)
        result["dataset"] = dataset
        result["factory"] = factory
        return result


    def run_analysis(self, datasets, factory, lr, lambda_):
        full = datasets[max(datasets.keys())]

        cost_each_epoch(
            full,
            lr,
            self.cfg.final_epochs,
            factory,
            lambda_
        )

        lambda_effect(
            full,
            self.cfg.fold_num,
            self.cfg.num_epochs,
            factory,
            lr,
            self.cfg.lambdas,
            seed=self.cfg.seed
        )

        size_effect_on_cost(
            datasets,
            self.cfg.fold_num,
            factory,
            self.cfg.num_epochs,
            lambda_,
            lr,
            seed=self.cfg.seed
        )

        size_effect_on_accuracy(
            datasets,
            factory,
            self.cfg.num_epochs,
            lambda_,
            lr,
            seed=self.cfg.seed
        )

def main():
    np.random.seed(42)

    # fetch dataset 
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 

    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_original.data.features 
    y = breast_cancer_wisconsin_original.data.targets 

    # Remove Sample_code_number (ID column) BEFORE converting to numpy
    if 'Sample_code_number' in X.columns:
        X = X.drop('Sample_code_number', axis=1)

    X = X.values
    y = y.values

    for i in range(X.shape[1]):
        col = X[:, i]
        median = np.nanmedian(col)
        col[np.isnan(col)] = median
        X[:, i] = col

    y = (y == 4).astype(int)

    x_train, y_train, x_test, y_test = global_split(X, y)

    config = Config()
    trainer = Trainer(config)
    exp = Experiment(trainer, config)

    print("Training models ...")
    raw = exp.run_single(
        x_train, y_train,
        x_test, y_test,
        False
    )

    feat = exp.run_single(
        x_train, y_train,
        x_test, y_test,
        True, raw["lr"], raw["lambda"]
    )

    print(
        f"RAW  ACC={raw['acc']:.4f} "
        f"LR={raw['lr']} "
        f"LAMBDA={raw['lambda']}"
    )

    print(
        f"FEAT ACC={feat['acc']:.4f} "
        f"LR={feat['lr']} "
        f"LAMBDA={feat['lambda']}"
    )

    best = feat if feat["acc"] > raw["acc"] else raw
    use_feat = best is feat

    print(
        "\nBest:",
        "Feature Engineered" if use_feat else "Raw"
    )

    print("Preparing datasets ...")
    datasets = {
        s: DatasetBuilder.build(
            x_train,
            y_train,
            x_test,
            y_test,
            s,
            use_feat
        )
        for s in config.sizes
    }

    full_size = len(x_train)
    datasets[full_size] = DatasetBuilder.build(
        x_train,
        y_train,
        x_test,
        y_test,
        full_size,
        use_feat
    )

    print("Running analysis ...")
    exp.run_analysis(
        datasets,
        best["factory"],
        best["lr"],
        best["lambda"]
    )

    print("Done.")


if __name__ == "__main__":
    main()