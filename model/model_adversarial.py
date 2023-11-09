import tensorflow as tf
import keras
import numpy as np


class Brain:
    def __init__(self, m_size, u_size, op_rate_shape, op_genre_shape):
        self.m_size= m_size
        self.u_size= u_size
        self.op_rate_shape= op_rate_shape
        self.op_genre_shape= op_genre_shape

        self.shared_op_size= 256

        self.make_model()

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def make_model(self):
        self.ip_m= keras.layers.Input(shape=(self.m_size,), name="input_m")
        self.ip_u= keras.layers.Input(shape=(self.u_size,), name="input_u")


        d_r1_m= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/2), activation="tanh")(self.ip_m)
        d_r1_u= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*3/2), activation="tanh")(self.ip_u)

        d_r2_m= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*5/3), activation="tanh")(d_r1_m)
        d_r2_u= keras.layers.Dense(units= int(min(self.m_size,self.u_size)*5/3), activation="tanh")(d_r1_u)


        shared_private= keras.layers.concatenate([d_r2_m, d_r2_u])

        d_p1= keras.layers.Dense(units= int((self.m_size+self.u_size)*2/3), activation="tanh")(shared_private)
        d_p2= keras.layers.Dense(units=1024, activation="sigmoid")(d_p1)
        self.shared_private_op= keras.layers.Dense(units=self.shared_op_size, activation="sigmoid")(d_p2)

        concated_1= keras.layers.concatenate([d_r2_m, d_r2_u])


        d_c1= keras.layers.Dense(units= int((self.m_size+self.u_size)/2), activation="tanh")(concated_1)
        d_c2= keras.layers.Dense(units=1024, activation="sigmoid")(d_c1)
        self.rating_private_op= keras.layers.Dense(units=self.shared_op_size, activation="sigmoid")(d_c2)

        shared_rating= keras.layers.concatenate([self.rating_private_op, self.shared_private_op])


        d_c5= keras.layers.Dense(units=self.op_rate_shape, activation="sigmoid", name= "out_rating")(shared_rating)

        concated_2= keras.layers.concatenate([d_r2_m, d_r2_u])


        d_d1= keras.layers.Dense(units= int((self.m_size+self.u_size)/3), activation="tanh")(concated_2)
        self.genre_private_op= keras.layers.Dense(units=self.shared_op_size, activation="sigmoid")(d_d1)


        shared_genre= keras.layers.concatenate([self.shared_private_op, self.genre_private_op])

        d_d5= keras.layers.Dense(units=self.op_genre_shape, activation="sigmoid", name= "out_genre")(shared_genre)


        self.model= keras.models.Model([self.ip_m, self.ip_u], [d_c5, d_d5])

        self.model.compile(loss="mse", optimizer="adam")



        #---------------------------------------------------------------------------------------------


        self.disc_op_shape=2

        discriminator_ip= keras.layers.Input(shape=(self.shared_op_size, ), name= "ip_discriminator")
        disc_d1= keras.layers.Dense(units= int(self.shared_op_size*2/3), activation="tanh")(discriminator_ip)
        disc_d2= keras.layers.Dense(units= 64, activation="tanh")(disc_d1)
        disc_op= keras.layers.Dense(units=self.disc_op_shape, activation="sigmoid")(disc_d2)
        
        self.disc_model= keras.models.Model(discriminator_ip, disc_op)
        self.disc_model.compile(loss="mse", optimizer="adam")


        self.shared_private_model_head=  keras.models.Model([self.ip_m, self.ip_u], shared_private)
        self.shared_private_model= keras.models.Model(shared_private, self.shared_private_op)



    
    def make_shared_private_train_model_wo_head(self):
        self.disc_model.trainable= False

        disc_and_private_model= keras.models.Sequential()
        disc_and_private_model.add(self.shared_private_model)
        disc_and_private_model.add(self.disc_model)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        disc_and_private_model.compile(loss="mse", optimizer=opt)

        return disc_and_private_model

    def fit_shared_private(self, xs, epochs=1, batch_size=64, verbose=2):
        print("  ==> Training shared private for adversarial...")
        shared_private_train_model= self.make_shared_private_train_model_wo_head()
        self.shared_private_model.trainable= True
        self.disc_model.trainable= False
        total_samples= len(xs[0])

        xs_sample= self.shared_private_model_head.predict(xs)
        ys_sample= np.zeros(total_samples*self.disc_op_shape).reshape(-1,self.disc_op_shape)

        shared_private_train_model.fit(xs_sample, ys_sample, epochs=epochs, batch_size=batch_size, verbose=verbose)


    def train_discriminator(self, xs, epochs=1, batch_size=64, verbose=2):
        print("  ==> Training Discriminator...")
        total_samples= len(xs[0])
        xs_samples, ys_samples= list(), list()

        # m1= keras.models.Model([self.ip_m, self.ip_u], self.shared_private_op)
        # xs_samples.extend(m1.predict(xs))
        # ys_samples.extend(np.ones(total_samples*self.shared_op_size).reshape(-1,self.shared_op_size))

        m2= keras.models.Model([self.ip_m, self.ip_u], self.rating_private_op)
        print(m2.predict(xs))
        xs_samples.extend(m2.predict(xs))
        ys_samples.extend(np.array([0,1]*total_samples, dtype=float).reshape(-1,self.disc_op_shape))

        m3= keras.models.Model([self.ip_m, self.ip_u], self.genre_private_op)
        xs_samples.extend(m3.predict(xs))
        ys_samples.extend(np.array([1,0]*total_samples, dtype=float).reshape(-1,self.disc_op_shape))


        xs_samples= np.array(xs_samples, dtype=float)
        ys_samples= np.array(ys_samples, dtype=float)

        xs_samples, ys_samples= self.unison_shuffled_copies(xs_samples, ys_samples)

        self.disc_model.trainable= True
        self.disc_model.fit(xs_samples, ys_samples, epochs=epochs, batch_size=batch_size, verbose=verbose)



    
    def fit_model(self, xs, ys, epochs=1, batch_size=64, verbose=2):
        print("  ==> Training main model")
        self.shared_private_model_head.trainable= True
        self.shared_private_model.trainable= True
        self.model.trainable= True
        self.model.fit(xs, ys, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def fit(self, xs, ys, epochs=1, batch_size=64, verbose=2):

        for e in range(epochs):
            print("\n")
            if epochs>1:
                print("     --- slide=", e+1, "---")

            self.train_discriminator(xs, epochs=1, batch_size=batch_size, verbose=verbose)
            self.fit_shared_private(xs, epochs=1, batch_size=batch_size, verbose=verbose)
            self.fit_model(xs, ys, epochs=1, batch_size=batch_size, verbose=verbose)

    def predict(self, xs, batch_size=64):
        res= self.model.predict(xs, batch_size=batch_size)
        return res

    def save(self, location, pred_only=False):
        location= location.split(".")[:-1]
        location= '.'.join(location)
        self.model.save(location+"_predictor.h5")
        if pred_only: return
        self.disc_model.save(location+"_discriminator.h5")
        shared_private_train_model= self.make_shared_private_train_model_wo_head()
        shared_private_train_model.save(location+"_shared_private_fit.h5")

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
    model.save("model_adversarial_3.h5", pred_only= True)
    # print("\n\nTOTAL PARAMS: ", model.count_params(), "\n\n")

    model.fit([m_enc, u_enc], [rate, genre], epochs=10)
