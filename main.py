from dataclasses import dataclass
import numpy as np

from plots import plot_3d_predictions
from generate_data import prep_dataset, global_split, generate_3d_classification_raw_data
from eval import lr_effect, lambda_effect, evaluate, cross_validation, cost_each_epoch, size_effect_on_accuracy, size_effect_on_cost 
from build_network import build_model

#  CONFIG 
@dataclass
class Config:
    fold_num: int = 3
    num_epochs: int = 25
    final_epochs: int = 100

    learning_rates: tuple = (0.1, 0.01, 0.001)
    lambdas: tuple = (0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)

    sizes: tuple = (500, 1000, 1500)


#  DATASET 
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


#  TRAINER 
class Trainer:
    def __init__(self, config):
        self.cfg = config

    def best_lr(self, dataset, factory):
        return lr_effect(
            dataset,
            self.cfg.learning_rates,
            self.cfg.fold_num,
            self.cfg.num_epochs,
            factory,
            plot=False
        )

    def best_lambda(self, dataset, factory, lr):
        best_cost = float("inf")
        best_lambda = self.cfg.lambdas[0]

        for lam in self.cfg.lambdas:
            _, cost = cross_validation(
                dataset.X_train,
                dataset.Y_train,
                self.cfg.fold_num,
                self.cfg.num_epochs,
                factory,
                lr,
                lam
            )

            if cost < best_cost:
                best_cost = cost
                best_lambda = lam

        return best_lambda

    def fit_model(self, dataset, factory, lr, lam):
        net = factory()

        net.fit(
            dataset.X_train,
            dataset.Y_train,
            self.cfg.final_epochs,
            lr,
            lam
        )

        return net

    def train_best(self, dataset, factory):
        lr = self.best_lr(dataset, factory)
        lam = self.best_lambda(dataset, factory, lr)

        net = self.fit_model(dataset, factory, lr, lam)

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
            "lambda": lam
        }


#  EXPERIMENT 
class Experiment:
    def __init__(self, trainer, config):
        self.trainer = trainer
        self.cfg = config

    def run_single(self, x_train, y_train, x_test, y_test, add_feat):
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
            return build_model(input_dim)

        result = self.trainer.train_best(dataset, factory)

        result["dataset"] = dataset
        result["factory"] = factory

        return result
    
    def run(self, x_train, y_train, x_test, y_test):
        raw = self.run_single(
            x_train, y_train,
            x_test, y_test,
            False
        )

        feat = self.run_single(
            x_train, y_train,
            x_test, y_test,
            True
        )

        return raw, feat

    def plot_best(self, result, surface):
        d = result["dataset"]

        plot_3d_predictions(
            d.X_test,
            d.Y_test,
            result["pred"],
            d.std,
            d.mean,
            surface
        )

    def run_analysis(self, datasets, factory, lr):
        full = datasets[max(datasets.keys())]

        cost_each_epoch(
            full,
            lr,
            self.cfg.final_epochs,
            factory
        )

        lambda_effect(
            full,
            self.cfg.fold_num,
            self.cfg.num_epochs,
            factory,
            lr,
            self.cfg.lambdas
        )

        lr_map = {k: lr for k in datasets.keys()}

        size_effect_on_cost(
            datasets,
            self.cfg.fold_num,
            lr_map,
            factory,
            self.cfg.num_epochs
        )

        size_effect_on_accuracy(
            datasets,
            lr_map,
            factory,
            self.cfg.num_epochs
        )


#  MAIN 
def main():
    np.random.seed(42)

    def surface(x, y):
        return (
            np.sin(x)
            + np.cos(y)
            + 0.15 * (x ** 2 + y ** 2)
            + 0.5 * np.sin(x * y / 4)
        )

    print("Generating data ...")

    X, y = generate_3d_classification_raw_data(
        4000,
        surface
    )

    x_train, y_train, x_test, y_test = global_split(X, y)

    config = Config()
    trainer = Trainer(config)
    exp = Experiment(trainer, config)

    print("Training models ...")

    raw, feat = exp.run(
        x_train,
        y_train,
        x_test,
        y_test
    )

    print(
        f"RAW  ACC={raw['acc']:.4f} "
        f"LR={raw['lr']} "
        f"L={raw['lambda']}"
    )

    print(
        f"FEAT ACC={feat['acc']:.4f} "
        f"LR={feat['lr']} "
        f"L={feat['lambda']}"
    )

    best = feat if feat["acc"] > raw["acc"] else raw
    use_feat = best is feat

    print(
        "\nBest:",
        "Feature Engineered"
        if use_feat else "Raw"
    )

    exp.plot_best(best, surface)

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
        best["lr"]
    )

    print("Done.")


if __name__ == "__main__":
    main()