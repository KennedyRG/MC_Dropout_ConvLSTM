from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, TimeDistributed, SpatialDropout2D
from tensorflow.keras.regularizers import l2

height, width = 12, 75

def mc_dropout_model():
    seq = Sequential()
    
    dropouts = [0.3, 0.35, 0.4, 0.45]

    for i, p in enumerate(dropouts):
        if i == 0:
            # Primera capa necesita input_shape
            seq.add(ConvLSTM2D(
                filters=16,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True,
                input_shape=(None, height, width, 1)
            ))
        else:
            seq.add(ConvLSTM2D(
                filters=16,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True,
                kernel_regularizer=l2(1e-4)
            ))

        # MC Dropout frame a frame
        seq.add(TimeDistributed(SpatialDropout2D(p)))

    # Capa final
    seq.add(Conv3D(
        filters=1,
        kernel_size=(1, 1, 1),
        activation='sigmoid',  # tus datos están normalizados entre 0 y 1
        padding='same'
    ))

    return seq

