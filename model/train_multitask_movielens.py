import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import helpers
import random
import numpy as np
import pandas as pd
import sys
import math
import help_movielens
from sklearn.metrics import mean_squared_error as mse

import model_fully_shared as neural_network
# import model_shared_private as neural_network
# import model_shared_private_3 as neural_network
# import model_adversarial as neural_network
# import model_adversarial_3 as neural_network
# import model_singletask_1 as neural_network







'''

movie_enc= helpers.load_pkl("../liv_data/objs/enc_movie.obj")
user_enc= helpers.load_pkl("../liv_data/objs/enc_user.obj")

'''





movie_enc= helpers.load_pkl("../objs/enc_movie.obj")
user_enc= helpers.load_pkl("../objs/enc_user.obj")
# movie_enc= helpers.load_pkl("../../Experiment_MMTF/enc_movie.obj")
# user_enc= helpers.load_pkl("../../Experiment_MMTF/enc_user.obj")

# save_model_name= "multitask_"+helpers.get_rand_str(5)+".h5"
save_model_name= "single_"+helpers.get_rand_str(5)+".h5"


genre_enc= helpers.load_pkl("../objs/genre_multihot.obj")
# genre_enc= helpers.load_pkl("../../Experiment_MMTF/genre_multihot.obj")



train_obj= helpers.load_pkl("../objs/u1.train.obj")
test_obj= helpers.load_pkl("../objs/u1.test.obj")
# train_obj= helpers.load_pkl("../../Experiment_MMTF/train_3000.obj")
# test_obj= helpers.load_pkl("../../Experiment_MMTF/test_3000.obj")


movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
user_shape= len(user_enc.get(list(user_enc.keys())[0]))
genre_shape= len(genre_enc.get(list(genre_enc.keys())[0]))



# batch_size= 64*128*2
batch_size=64

xyz=pd.read_csv('../objs/test.csv')




def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def shuffle_single_epoch(ratings):
    data_copied= ratings.copy()
    random.shuffle(data_copied)
    return data_copied


def normalize(rate):
    return rate/5
def de_normalize(rate):
    return rate*5




def copy_to_fix_shape(req_len, *elements):
    if len(elements)==0: 
        print("not found")
        return
    len_of_each= len(elements[0])
#     print(len_of_each)
#     print(elements)
#     print(req_len)
    for e in elements:
        if not len_of_each == len(e):
            raise Exception("All elements are not of same shape.. unable to fix shape...!!!")


    # if has elements more than required len
    if len_of_each>req_len:
        elements= list(elements)
        for item_idx in range(len(elements)):
            elements[item_idx]= elements[item_idx][:req_len]


    # if has elements less than required len
    while len(elements[0]) < req_len:
        r_ind= random.randint(0, len_of_each-1)
        for item_idx in range(len(elements)):
            elements[item_idx].append(elements[item_idx][r_ind])
    

    return elements



def get_nth_batch(ratings, n, batch_size= batch_size, take_entire_data= False):
    users= []
    movies= []
    rates= []
    genre= []
    if take_entire_data:
        slice_start= 0
        slice_end= len(ratings)
#         print(slice_end)
    else:
        if (n+1)*batch_size>len(ratings):
            if n*batch_size>=len(ratings):
                print("OUT OF RANGE BATCH ID")
                raise Exception("ERROR!!! No data to take...")
        slice_start= n*batch_size
        slice_end= (n+1)*batch_size
    cnt=0
    for user_id, movie_id, rate in ratings[slice_start: slice_end]:
#         print(user_enc.get(user_id))
#         print(len(user_enc.get(user_id)))
#         print(len(movie_enc.get(movie_id)))
#         print("user: "user_id, , " idx: ", cnt,end='\r')
#         cnt+=1
        if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None or genre_enc.get(movie_id) is None:
#         if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None or genre_enc.get(movie_id) is None:
            continue
        users.append(user_enc.get(user_id))
        movies.append(movie_enc.get(movie_id))
        rates.append(normalize(rate))

        genre.append(genre_enc.get(movie_id))

    if len(users)==0:
        return movies, users, rates, genre
    if not take_entire_data:
        users, movies, rates, genre= copy_to_fix_shape(batch_size, users, movies, rates, genre)
#     print("1st se phele checkpoint")
    users= np.array(users)
    movies= np.array(movies)
    rates= np.array(rates)
    genre= np.array(genre)


    return movies, users, rates, genre


extra_epoch=0
def train(model, data, test_data= None, no_of_epoch= 32, recurs_call=False):
    global extra_epoch
    total_batches_train= math.ceil(len(data)/batch_size)
    print("total_batches_train = ",total_batches_train)
    for epoch in range(no_of_epoch):
        if recurs_call:
            print("\n\n---- EXTRA EPOCH: ", extra_epoch, "------\n\n")
            extra_epoch+=1
        else:
            print("\n\n---- EPOCH: ", epoch, "------\n\n")
        data= shuffle_single_epoch(data)
        for batch_id in range(total_batches_train):
            print("Epoch: ", epoch+1, " Batch: ", batch_id,end='\r')
            movies, users, rates, genre= get_nth_batch(data, batch_id)
            model.fit([movies, users], [rates, genre], batch_size=batch_size, epochs=1, verbose=0)
        if test_data is not None:
            print("enterted testing")
            test(model, test_data, take_entire_data=True, save=True)

    while True and not recurs_call:
        some_last_rmse= all_rmse[-7:]
        best_rmse= min(all_rmse)
        if best_rmse not in some_last_rmse:
            break
        best_in_last= False
        for l in some_last_rmse:
            if abs(l-best_rmse)<0.01:
                best_in_last= True
        if not best_in_last: break

        train(model, data, test_data=test_data, no_of_epoch=1, recurs_call=True)



lest_rmse= float("inf")
all_rmse= list()
def test(model, data, save= True, take_entire_data= True):
#     if take_entire_data:
# #         print("1st checkkpoint")
#         movies, users, res_true, _= get_nth_batch(data, 0, take_entire_data=take_entire_data)
# #         print("2nd checkpoint")
#         res_pred,_= model.predict([movies, users], batch_size=batch_size)
#         res_true= np.array(res_true)
#         res_pred= np.array(res_pred).reshape(-1)
#         print(len(res_true))
#         assert len(res_true)==len(res_pred)
    if take_entire_data:
        print("test entered")
#         movies, users, res_true, _, _= get_nth_batch(data, 0, take_entire_data=take_entire_data)
        rmse_true, rmse_pred= np.array([]), np.array([])
        arr=help_movielens.fun()
        final_ans=0.0
        final_recall=0.0
        final_recall_k=0.0
        ndcg_ans=0.0
        n1=len(arr)
        print(n1)
#         arr_ndcg=[]
        list_of_movie_pred=[]
        list_of_movie_true=[]
        cnt=0
#         arr=arr.sort()
#         n1=3000
        user=[]
#         global epoch
        for i in arr:
#             print("user: ",cnt)
#             if cnt==3000:
#                 break
#             cnt+=1
            users=[]
            movies=[]
            rates=[]
            df=xyz[xyz['userId']==i]
#             print(df)

            for index,movieid in df.iterrows():
                user_id=int(movieid['userId'])
                movie_id=int(movieid['movieId'])
                rate=movieid['rating']
                if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None:
                    continue
                users.append(user_enc.get(user_id))
                movies.append(movie_enc.get(movie_id))
                rates.append(normalize(rate))
            if len(users)==0:
                continue
            users= np.array(users)
            movies= np.array(movies)
            res_pred,_= model.predict([movies, users], batch_size=batch_size)
            res_true= np.array(rates)
            res_pred= np.array(res_pred).reshape(-1)
#             print(len(res_true))
#             print(len(res_pred))
#             k=3.5
            res_true= de_normalize(res_true)
            res_pred= de_normalize(res_pred)
            rmse_true= np.concatenate([rmse_true, res_true])
            rmse_pred= np.concatenate([rmse_pred, res_pred])
#             user.append([res_true,res_pred])
            
            dict_true=[]
            dict_pred=[]
            for j in range(0,len(res_true)):
                dict_true.append([res_true[j],j])
                dict_pred.append([res_pred[j],j])
            sorter = lambda x: (-x[0], x[1])
#             print(dict_true)
#             print(dict_pred)
            dict_true = sorted(dict_true, key=sorter)
            dict_pred = sorted(dict_pred, key=sorter)
            rel=0
            for j in range(0,len(dict_pred)):
                x=dict_pred[j][1]
                y=dict_true[j][1]
                if x==y:
                    rel+=1
            dict_pred=dict_pred[:5]
            dict_true=dict_true[:5]
#             print(dict_true)
#             print(dict_pred)
            res=0.0
            gtp=0
            recall=0.0
            arr_ndcg=[]
            
            for j in range(0,len(dict_pred)):
#                 movie_pred.append(dict_pred[j][1])
#                 movie_true.append(dict_true[j][1])
                x=dict_pred[j][1]
                y=dict_true[j][1]
                if x==y:
                    arr_ndcg.append(1)
                    gtp+=1
                    res+=float(gtp)/(j+1)
                    if gtp==1:
                        recall=1.0/(j+1)
                else:
                    arr_ndcg.append(0)
            
            if gtp==0:
                continue
            # MRR
            final_recall+=recall
            
            # MAP
            res=res/gtp
            final_ans+=res
            
            #RECALL@K
            rel=float(gtp)/rel
            final_recall_k+=rel
            
            ndcg_ans+=ndcg_at_k(arr_ndcg, 5)
            
        final_recall_k=final_recall_k/n1    
        final_ans=final_ans/n1
        final_recall=final_recall/n1
        final_ndcg_ans=ndcg_ans/n1
#         save_file_per_epoch=str(epoch)
#         np.save(save_file_per_epoch,user)
        print("MAP: ",final_ans," MRR: ",final_recall,"R: ",final_recall_k,"NDCG: ",final_ndcg_ans)
    else:
        total_batches_test=int(len(data)/batch_size)
        res_true, res_pred= np.array([]), np.array([])
        print("total_batches_test = ",total_batches_test)
        for batch_id in range(total_batches_test+1):
            movies, users, rates, _= get_nth_batch(data, batch_id)
            if len(rates)==0:
#                 print("not found batch_id = ",batch_id)
                continue
            print(" Batch: ", batch_id,end='\r')
            pred, _= model.predict([movies, users], batch_size=batch_size)
            pred= pred.reshape(-1)
            assert len(rates)==len(pred)
            res_true= np.concatenate([res_true, rates])
            res_pred= np.concatenate([res_pred, pred])
#     y_true= de_normalize(res_true)
#     y_pred= de_normalize(res_pred)

#     for x in range(min(len(y_true), 200)):
#         print(y_true[x], " : ", y_pred[x])

#     rmse= calc_rms(y_true, y_pred)
#     y_pred= np.array([round(x) for x in y_pred])
#     rmse_n= calc_rms(y_true, y_pred)
#     print("rmse: ", rmse, " rmse_norm: ", rmse_n)
#     global lest_rmse
#     if save and lest_rmse>rmse:
#         lest_rmse= rmse
#         helpers.save_model(model, save_filename=save_model_name)
    
#     all_rmse.append(rmse)
    rmse= calc_rms(rmse_true, rmse_pred)
    all_rmse.append(rmse)
#     y_pred= np.array([round(x) for x in y_pred])
#     rmse_n= calc_rms(y_true, y_pred)
    print("rmse: ", rmse)
    global lest_rmse
    if save and lest_rmse>rmse:
        lest_rmse= rmse
        helpers.save_model(model, save_filename=save_model_name)



def calc_rms(t, p):
    return mse(t, p, squared=False)


def train_test_ext(train_obj, test_obj):
    model= neural_network.make_model(movie_shape, user_shape, 1, genre_shape)
    train(model, data=train_obj, test_data=test_obj)
    test(model, data= test_obj, save= False, take_entire_data=True)


def test_loaded_model():
    model= sys.argv[2]
    import keras
    model= keras.models.load_model(model)
    test(model, test_obj, save=False, take_entire_data=True)
    # test(model, train_obj[30000:40000], save=False, take_entire_data=True)

    exit()


def print_metadata():
    print("Movie_shape: ", movie_shape)
    print("User_shape: ", user_shape)

    print("\ngenre_shape: ", genre_shape)

    print("batch_size:", batch_size)

    print("Saved_model_name: ", save_model_name)

    print("\n\n")



if __name__=="__main__":

    if len(sys.argv)==3 and sys.argv[1]=='-t' and sys.argv[2].split('.')[-1]=='h5':
        test_loaded_model()



    print_metadata()

    # train_obj= train_obj[:100]
    # test_obj= test_obj[:100]


    train_test_ext(train_obj, test_obj)


