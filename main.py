from train import train_model




if __name__ == "__main__":
    train_model(
        learning_rate=0.01,
        decay_learning_rate=1e-8,
        all_trainable=False,
        model_weights_path=None,
        no_class=197,
        batch_size=32,
        epoch=200
    )
