import tensorflow as tf
import keras
import numpy as np

def my_loss_fn(y_true, y_pred):
#     print("enter")
#     print(y_true.shape," ")
#     print("exit")
    y_true = tf.cast(y_true, dtype=tf.float32)
    n=y_true.shape[0]
    dict_true=[]
    dict_pred=[]
    for i in range(0,n):
        x=y_true[i].numpy()
        y=y_pred[i].numpy()
        dict_true.append([x[0],i])
        dict_pred.append([y[0],i])
    sorter = lambda x: (-x[0], x[1])
    dict_true = sorted(dict_true, key=sorter)
    dict_pred = sorted(dict_pred, key=sorter)
    arr_pred=[]
    arr_true=[]
    for j in range(0,len(dict_pred)):
        x=dict_pred[j][1]
        y=dict_true[j][1]
        arr_true.append(0.0)
        if x==y:
            arr_pred.append(1.0)
        else:
            arr_pred.append(0.0)
    arr_pred=np.array(arr_pred)
    arr_true=np.array(arr_true)
#     arr_pred=np.zeros(n)
#     arr_true=np.zeros(n)
    y_true=tf.convert_to_tensor(arr_true, dtype=tf.float32)
    y_pred=tf.convert_to_tensor(arr_pred, dtype=tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

losses = {
	"out_rating":my_loss_fn ,
	"out_genre": "MSE",
}

def make_model(m_size, u_size, op_rating_size, op_genre_size):

    ip_m= keras.layers.Input(shape=(m_size,), name="input_m")
    ip_u= keras.layers.Input(shape=(u_size,), name="input_u")

    d_r1_m= keras.layers.Dense(units= int(min(m_size,u_size)*3/2), activation="tanh")(ip_m)
    d_r1_u= keras.layers.Dense(units= int(min(m_size,u_size)*3/2), activation="tanh")(ip_u)

    d_r2_m= keras.layers.Dense(units= int(min(m_size,u_size)*5/3), activation="tanh")(d_r1_m)
    d_r2_u= keras.layers.Dense(units= int(min(m_size,u_size)*5/3), activation="tanh")(d_r1_u)

    concated_1= keras.layers.concatenate([d_r2_m, d_r2_u])
    d_c1= keras.layers.Dense(units= int((u_size+m_size)/2), activation="tanh")(concated_1)
    d_c2= keras.layers.Dense(units=1024, activation="sigmoid")(d_c1)
    d_c3= keras.layers.Dense(units=128, activation="sigmoid")(d_c2)
    d_c4= keras.layers.Dense(units=op_rating_size, activation="sigmoid", name= "out_rating")(d_c3)

    concated_2= keras.layers.concatenate([d_r2_m, d_r2_u])
    d_d1= keras.layers.Dense(units= int((u_size+m_size)/3), activation="tanh")(concated_2)
    d_d3= keras.layers.Dense(units=128, activation="sigmoid")(d_d1)
    d_d4= keras.layers.Dense(units=op_genre_size, activation="sigmoid", name= "out_genre")(d_d3)

    model= keras.models.Model([ip_m, ip_u], [d_c4, d_d4])
#     model= keras.models.Model([ip_m, ip_u], [d_c4])
#     print(d_c4)
    model.compile(loss="MSE", optimizer="adam")
#     model.compile(loss=losses, optimizer="adam",run_eagerly=True)
#     model.compile(loss=losses, optimizer="adam")
    return model

if __name__ == "__main__":
    m_size, u_size, op_1_size, op_2_size= 1220, 1243, 1, 19
    model= make_model(m_size, u_size, op_1_size, op_2_size)
    model.save("multitask.h5")
