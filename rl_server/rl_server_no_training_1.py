from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import base64
import urllib
import sys
import os
import json
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import tensorflow as tf
import time
import a3c

import random

# pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from sklearn import preprocessing
import torch 




port_id  = 8333
S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE_MAP = {300:1,750:2,1200:3,1850:4,2850:5,4300:6} # Kbps

MEMORY_DIC =  {300: 1, 750: 5, 1200: 3 ,1850: 10,2850: 15,2850: 20,4300: 30}
CLIENT_MEMORY =  {1: 1000, 2: 3000, 3: 6000 , 4:32000}

CLIENT_1_MOS =   {300: 3, 750: 4, 1200: 5 ,1850:6, 2850:6, 4300:6}
CLIENT_2_MOS =   {300: 2, 750: 3, 1200: 4 ,1850:5, 2850:6, 4300:6}
CLIENT_3_MOS =   {300: 1, 750: 2, 1200: 3 ,1850:4, 2850:6, 4300:6}
CLIENT_4_MOS =   {300: 1, 750: 1, 1200: 2 ,1850:2, 2850:3, 4300:5}

CLIENT_1_MOS_VIDEO =   {3:300,   4:750,  5:1200,   6:1850, 6:2850, 6:4300}
CLIENT_2_MOS_VIDEO =   {2:300,   3:750,  4:1200,   5:1850, 6:2850, 6:4300}
CLIENT_3_MOS_VIDEO =   {1:300,   2:750,  3:1200,   4:1850, 6:2850, 6:4300}
CLIENT_4_MOS_VIDEO =   {1:300,   1:750,  2:1200,   2:1850, 4:2850, 5:4300}

#6 = No difference
#5 = Excellent
#4 = Very good
#3 = Good
#2 = Fair
#1 = Poor

CLIENT_LIST = [1,2,3,4]  
CLIENT_ID = 1

BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
BITRATE_REWARD_MAP_C1 = {0: 0, 300:15 , 750: 20, 1200: 12, 1850: 10, 2850: 3, 4300: 1}


M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
LOG_FILE_PER =  './results/log_per'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size download_time reward
# NN_MODEL = None
NN_MODEL = '../rl_server/results/pretrain_linear_reward.ckpt'
PER_MODEL = './model_central_training_2.pt'
cpu_load = 0
cpu_ini =  70 
memory_load = 0
memory_ini  = 50
memory_queue = []

#6 = No difference
#5 = Excellent
#4 = Very good
#3 = Good
#2 = Fair
#1 = Poor


tf.compat.v1.disable_eager_execution()

# video chunk sizes
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]


def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.sess = input_dict['sess']
            self.log_file = input_dict['log_file']
            self.actor = input_dict['actor']
            self.critic = input_dict['critic']
            self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            self.a_batch = input_dict['a_batch']
            self.r_batch = input_dict['r_batch']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            print(post_data)

            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print("Summary: "+ post_data)
            else:
                # option 1. reward for just quality
                # reward = post_data['lastquality']
                # option 2. combine reward for quality and rebuffer time
                #           tune up the knob on rebuf to prevent it more
                # reward = post_data['lastquality'] - 0.1 * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
                # option 3. give a fixed penalty if video is stalled
                #           this can reduce the variance in reward signal
                
                #reward = reward_quality  
                reward = post_data['lastquality'] - 10 * ((post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) > 0)

                # option 4. use the metric in SIGCOMM MPC paper
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
                last_quality = post_data['lastquality']
                reward_quality = VIDEO_BIT_RATE[post_data['lastquality']]
                bitrate = VIDEO_BIT_RATE[post_data['lastquality']]
                vmos = CLIENT_1_MOS[bitrate]
               # if self.input_dict['video_chunk_coount'] 
               #     memory_queue

                    
               # if  vmos != 3 :
               #     reward_quality =   (reward_quality * -1)
               # print('reward_quality:'+str(reward_quality))   

                # --linear reward--
                #reward = reward_quality / M_IN_K \
                #        - REBUF_PENALTY * rebuffer_time / M_IN_K \
                #        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                #                                  self.input_dict['last_bit_rate']) / M_IN_K
              
                print('TOTAL REWARD:'+str(reward))     

                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))   
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))

                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']

                # retrieve previous state
                if len(self.s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(self.s_batch[-1], copy=True)

                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
                self.input_dict['video_chunk_coount'] += 1

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                next_video_chunk_sizes = []
                for i in range(A_DIM):
                    next_video_chunk_sizes.append(get_chunk_size(i, self.input_dict['video_chunk_coount']))

                # this should be S_INFO number of terms
                try:
                    state[0, -1] = VIDEO_BIT_RATE[post_data['lastquality']] / float(np.max(VIDEO_BIT_RATE))
                    state[1, -1] = post_data['buffer'] / BUFFER_NORM_FACTOR
                    state[2, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K  # kilo byte / ms
                    state[3, -1] = float(video_chunk_fetch_time) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                    state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                    state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
                except ZeroDivisionError:
                    # this should occur VERY rarely (1 out of 3000), should be a dash issue
                    # in this case we ignore the observation and roll back to an eariler one
                    if len(self.s_batch) == 0:
                        state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        state = np.array(self.s_batch[-1], copy=True)

                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, download_time, reward
                self.log_file.write(str(time.time()) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(rebuffer_time / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                print(bit_rate)

                vmos = getVMOS(CLIENT_ID,bitrate)
                print('vmos predicted bitrate'+str(vmos))
                if vmos >= 4: 
                    id_randon = 3

                    bit_rate = VIDEO_BIT_RATE_MAP[getBitrate(CLIENT_ID,id_randon)]
                    print('replacing bitrate'+str(bit_rate))


                # send data to html side
                send_data = str(bit_rate)

                end_of_video = False
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = "REFRESH"
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_coount'] = 0
                    self.log_file.write('\n')  # so that in the log we know where video ends

                bitrate = VIDEO_BIT_RATE[post_data['lastquality']]
                memory_consup = MEMORY_DIC[bitrate]  
                buffer_size = round(post_data['buffer'])
                memory_load =   round(memory_consup) *  round(buffer_size)
                percent_memory=  round(memory_load) / CLIENT_MEMORY[CLIENT_ID]
                rebuffer_time_log = str(rebuffer_time / M_IN_K)
                vmos = CLIENT_1_MOS[bitrate]
                rmos = 1
                bandwidth = round(post_data['bandwidthEst'])

                if percent_memory > 0.8:
                     rmos = 0


                with open(LOG_FILE_PER, 'a') as log_per:   
                    line = str(self.input_dict['video_chunk_coount'])+';'+str(CLIENT_ID)+';'+str(CLIENT_ID)+';1;1;'+str(bandwidth)+';'+str(VIDEO_BIT_RATE[post_data['lastquality']])+';'+str(buffer_size)+';'+str(rebuffer_time)+';'+'6'+';'+'10'+';'+str(memory_load)+';'+str(percent_memory)+';'+str(vmos)+';'+str(rmos)
                    log_per.write(line) 
                    log_per.write('\n') 
                    log_per.flush()

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode('utf-8'))

                # record [state, action, reward]
                # put it here after training, notice there is a shift in reward storage

                if end_of_video:
                    self.s_batch = [np.zeros((S_INFO, S_LEN))]
                else:
                    self.s_batch.append(state)

        def do_GET(self):
            print(str(sys.stderr) + 'GOT REQ')
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');".encode('utf-8'))

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port_id=port_id, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)


    with tf.compat.v1.Session()as sess, open(log_file_path, 'w') as log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver =  tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        init_action = np.zeros(A_DIM)
        init_action[DEFAULT_QUALITY] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [init_action]
        r_batch = []

        train_counter = 0

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'sess': sess, 'log_file': log_file,
                      'actor': actor, 'critic': critic,
                      'saver': saver, 'train_counter': train_counter,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_coount': video_chunk_count,
                      's_batch': s_batch, 'a_batch': a_batch, 'r_batch': r_batch}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)
        with open(LOG_FILE_PER, 'w') as log_per:  
            header =  'id;user;device;video_type;complexity;bandwidth;quality;buffersize;Rebuffer;screen_size;resolution;memory_load;percent_memory;vmos;rmos'  
            log_per.write(header) 
            log_per.write('\n') 
            log_per.flush()


        server_address = ('10.0.0.4', int(port_id))
        httpd = server_class(server_address, handler_class)
        print('Listening on port ' + str(port_id))
        httpd.serve_forever()

def getVMOS(client_id,bitrate):
    if client_id == 1:
        vmos = CLIENT_1_MOS[bitrate]
    elif client_id == 2:
        vmos = CLIENT_2_MOS[bitrate]
    elif client_id == 3:
        vmos = CLIENT_3_MOS[bitrate]
    else : 
        vmos = CLIENT_4_MOS[bitrate]
    return vmos


def getBitrate(client_id,vmos):
    if client_id == 1:
        vmos = CLIENT_1_MOS_VIDEO[vmos]
    elif client_id == 2:
        vmos = CLIENT_2_MOS_VIDEO[vmos]
    elif client_id == 3:
        vmos = CLIENT_3_MOS_VIDEO[vmos]
    else : 
        vmos = CLIENT_4_MOS_VIDEO[vmos]
    return vmos

def main():
    if len(sys.argv) == 3:
        trace_file = sys.argv[1]
        port_id  = sys.argv[2]
        run(port_id=port_id, log_file_path=LOG_FILE + '_RL_' + trace_file)
    else:
        port_id  = sys.argv[2]
        run(port_id=port_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)