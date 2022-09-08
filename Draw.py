from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model(input('model_file:'))
z_size = model.layers[1].weights
z_size = z_size[0].shape[0]

while True:
    z = np.random.normal(0, 1, (1, z_size))
    picture = model.predict(z)[0]
    plt.imshow(picture)
    plt.show()
