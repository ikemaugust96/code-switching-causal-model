from universality_eval import run_universality_experiments

if __name__ == "__main__":
    run_universality_experiments(
        max_pairs=2,
        skip_lr=True,
        gru_epochs=2,
        gru_max_train_samples=100000,
        gru_max_test_samples=20000,
    )