import tensorflow as tf
import keras


class Brain:
    def __init__(self, m_size, u_size, op_rate_shape, op_genre_shape):
        self.m_size= m_size
        self.u_size= u_size
        self.op_rate_shape= op_rate_shape
        self.op_genre_shape= op_genre_shape

        self.make_model()

    def make_model(self):
        ip_m= keras.layers.Input(shape=(self.m_size,), name="input_m")
        ip_u= keras.layers.Input(shape=(self.u_size,), name="input_u")

        d_r1_m= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/2), activation="tanh")(ip_m)
        d_r1_u= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/2), activation="tanh")(ip_u)

        d_drop1_m= keras.layers.Dropout(rate=0.2)(d_r1_m)
        d_drop1_u= keras.layers.Dropout(rate=0.2)(d_r1_u)

        d_r2_m= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/5), activation="tanh")(d_drop1_m)
        d_r2_u= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/5), activation="tanh")(d_drop1_u)


        d_r2_m= keras.layers.Dropout(rate=0.2)(d_r2_m)
        d_r2_u= keras.layers.Dropout(rate=0.2)(d_r2_u)

        d_r3_m= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*5/7), activation="tanh")(d_r2_m)
        d_r3_u= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*5/7), activation="tanh")(d_r2_u)

        concated= keras.layers.concatenate([d_r1_m, d_r1_u])

        concated= keras.layers.Dropout(rate=0.2)(concated)
        shared_private= keras.layers.Dense(units= int(min(self.m_size,self.u_size)/2), activation="tanh")(concated)
        shared_private= keras.layers.Dropout(rate=0.2)(shared_private)
        shared_private= keras.layers.Dense(units=int(min(self.m_size,self.u_size)*5/7), activation="sigmoid")(shared_private)

        task_rate_conc= keras.layers.concatenate([d_r3_m, shared_private])
        task_genre_conc= keras.layers.concatenate([d_r3_u, shared_private])

        d_rate_0= keras.layers.Dense(units=1024, activation="sigmoid")(task_rate_conc)
        d_rate_1= keras.layers.Dense(units=128, activation="sigmoid")(d_rate_0)
        d_rate_2= keras.layers.Dense(units=self.op_rate_shape, activation="sigmoid", name= "out_rating")(d_rate_1)

        d_genre_0= keras.layers.Dense(units=128, activation="sigmoid")(task_genre_conc)
        d_genre_1= keras.layers.Dense(units=128, activation="sigmoid")(d_genre_0)
        d_genre_2= keras.layers.Dense(units=self.op_genre_shape, activation="sigmoid", name= "out_genre")(d_genre_1)

        self.model= keras.models.Model([ip_m, ip_u], [d_rate_2, d_genre_2])

        self.model.compile(loss="mse", optimizer="adam")

    
    def fit(self, xs, ys, epochs=1, batch_size=64, verbose=2):
        self.model.fit(xs, ys, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, xs, batch_size=64):
        res= self.model.predict(xs, batch_size=batch_size)
        return res

    def save(self, location):
        self.model.save(location)

    def count_params(self):
        return self.model.count_params()


        

def make_model(m_size, u_size, op_rate_shape, op_genre_shape):
    model= Brain(m_size, u_size, op_rate_shape, op_genre_shape)
    return model
    




if __name__ == "__main__":
    import numpy as np

    m_size, u_size= 1220, 1243
    rate_shape, genre_shape= 1, 19

    sample_count= 100

    m_enc= np.random.random(sample_count*m_size).reshape(-1, m_size)
    u_enc= np.random.random(sample_count*u_size).reshape(-1, u_size)

    rate= np.random.random(sample_count*rate_shape).reshape(-1, rate_shape)
    genre= np.random.random(sample_count*genre_shape).reshape(-1, genre_shape)

    model= make_model(m_size, u_size, rate_shape, genre_shape)
    model.save("model_shared_private.h5")
    print("\n\nTOTAL PARAMS: ", model.count_params(), "\n\n")

    model.fit([m_enc, u_enc], [rate, genre], epochs=10)
