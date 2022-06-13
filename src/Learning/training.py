from b_cloning.b_cloning import behaviouralCloning_training

if __name__ == '__main__':
    print("Running as main")

    epochs = 180
    batch_size = 95
    weight_decay = 3e-7

    behaviouralCloning_training("followDummy_fixed_2", "LSTM_largerBaseCNN_follow_2", "LSTM_largerBaseCNN", epochs=epochs, batch_size=batch_size, weight_decay=weight_decay, training_method='eeVel_aux')