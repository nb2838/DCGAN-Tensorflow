import matplotlib.pyplot as plt
import numpy as np


def make_grid(batch, elem_num, path ='None',figsize=(10,10)):
    """
    Takes first elem_num elements of the data and plots them in a grid
    Assumes elem num is squared so that we get a proper grid
    """
    side = int(np.sqrt(elem_num))
    fig = plt.figure(figsize=figsize)
    for i in range(elem_num):
        ax = fig.add_subplot(side, side, i + 1)
        ax.imshow(batch[i].numpy().astype("uint8"))
        ax.axis("off")
        
    if path: 
        fig.savefig(path)
    plt.close()


def make_batch_generator(data, batch_size=128):
    """
    Returns a generator to be used for training
    """

    while True:
        data = data[np.random.permutation(len(data))]
        i = 0
        while i + batch_size < len(data):
            
            yield data[i:i+batch_size]
            
                         





