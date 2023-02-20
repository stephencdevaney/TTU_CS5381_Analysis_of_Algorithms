from fog_env import Offload
from RL_brain import DeepQNetwork
import numpy as np
import pandas as pd
import os.path
import random
# import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

dropped_task_ratio=[]
rdt_vs_tap = False
rdt_vs_nmd = False
rdt_vs_td = False

def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):

    # still use reward, but use the negative value
    penalty = - max_delay * 2

    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay

    return reward


def train(iot_RL_list, NUM_EPISODE):

    RL_step = 0
    for i in range(NUM_EPISODE):
        dropped_task_ratio.append(0)
    for iot_index in range(NUM_IOT):
        prob.append(random.uniform(0, 1))
    for episode in range(NUM_EPISODE):

        print('Episode: ',episode)
        #print(iot_RL_list[0].epsilon)
        # BITRATE ARRIVAL
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # =================================================================================================
        # ========================================= DRL ===================================================
        # =================================================================================================

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for iot_index in range(env.n_iot):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([env.n_iot])
            for iot_index in range(env.n_iot):

                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count, action_all[iot_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            for iot_index in range(env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index,:])
            
                #prob.append(random_pick(iot_RL_list[iot_index], task_prob))
                

            process_delay = env.process_delay
            #print('Delay: ', process_delay)
            unfinish_indi = env.process_delay_unfinish_ind

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for iot_index in range(env.n_iot):

                history[env.time_count - 1][iot_index]['observation'] = observation_all[iot_index, :]
                history[env.time_count - 1][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[env.time_count - 1][iot_index]['action'] = action_all[iot_index]
                history[env.time_count - 1][iot_index]['observation_'] = observation_all_[iot_index]
                history[env.time_count - 1][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index,:])

                update_index = np.where((1 - reward_indicator[:,iot_index]) * process_delay[:,iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        iot_RL_list[iot_index].store_transition(history[time_index][iot_index]['observation'],
                                                                history[time_index][iot_index]['lstm'],
                                                                history[time_index][iot_index]['action'],
                                                                reward_fun(process_delay[time_index, iot_index],
                                                                           env.max_delay,
                                                                           unfinish_indi[time_index, iot_index]),
                                                                history[time_index][iot_index]['observation_'],
                                                                history[time_index][iot_index]['lstm_'])
                        iot_RL_list[iot_index].do_store_reward(episode, time_index,
                                                               reward_fun(process_delay[time_index, iot_index],
                                                                          env.max_delay,
                                                                          unfinish_indi[time_index, iot_index]))
                        
                        iot_RL_list[iot_index].do_store_delay(episode, time_index,
                                                              process_delay[time_index, iot_index])
                
                        #print("Delay is:",delayy)
                        #delay.append(delayy)
                        reward_indicator[time_index, iot_index] = 1
                 
                                        
                    delay[iot_index] = process_delay[time_index, iot_index]
                    tasks[iot_index]= sum(unfinish_indi)/len(unfinish_indi)
                    
                        #print(delay)
                    
                        #print(observation)
                        #print('Process delay array: ', process_delay)
            # ADD STEP (one step does not mean one store)
                    
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            #print('Observation: ', observation_all)
            lstm_state_all = lstm_state_all_
            


            # CONTROL LEARNING START TIME AND FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            # GAME ENDS
            if done:
                #return observation_all, delay
                break
        dropped_task_ratio[episode] = (env.drop_trans_count + env.drop_fog_count + env.drop_iot_count)/env.total_count
        #  =================================================================================================
        #  ======================================== DRL END=================================================
        #  =================================================================================================
def plot_delays(delay, prob):
    
    print('Delay is: ', delay)
    print(len(delay))
    delay.sort()
    delay =[i/(np.max(delay)*1.25) for i in delay]
    print('Norm sorted Delay is: ', delay)
    print(len(delay))
    #df = pd.DataFrame({'delay': delay, 'delay2':delay})
            
    #df['delay_rolling_avg'] = df.delay.rolling(10).mean()
    #df['delay_rolling_avg2'] = df.delay2.rolling(14).mean()
    
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter1d

    ysmoothed = gaussian_filter1d(delay, sigma=2)
    deadlines=[0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4]
        #print(self.loss_his)
    #plt.plot(np.arange(len(delay)), delay)
    
    plt.plot(np.arange(len(delay)),ysmoothed,marker='x',color='green')
    plt.ylabel('Average Delay')
    plt.xlabel('No. of Mobile Devices')
    plt.show()
    
    
    plt.plot(deadlines,ysmoothed,marker='x',color='green')
    plt.ylabel('Average Delay')
    plt.xlabel('Task Deadline')
    plt.show()
    
    
    
    print('Task arrival probability',prob)
    print(len(prob))
    #prob.sort()
    #prob =[np.average(i) for i in prob]
    prob =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98]
    ysmoothed2 = gaussian_filter1d(delay, sigma=2)
    #print(self.loss_his)
    #plt.plot(np.arange(len(delay)), delay)
    plt.plot(prob,ysmoothed2,marker='x',color='green')
    plt.ylabel('Average Delay')
    plt.xlabel('Task Arrival Probability')
    plt.show()

def plot_cost():
    import matplotlib.pyplot as plt
        
    avg_cost =[5*(i/(np.median(avg_cost)*NUM_IOT)) for i in avg_cost]
    df = pd.DataFrame({'acost': avg_cost})
            
    df['cost_rolling_avg'] = df.acost.rolling(190).mean()
    
    from scipy.ndimage.filters import gaussian_filter1d

    ysmoothed = gaussian_filter1d(df['cost_rolling_avg'], sigma=2)
    
    plt.plot(np.arange(len(avg_cost)), ysmoothed, color='green', marker='v', label= 'lr = 0.01')
    plt.ylabel('Cost')
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(np.arange(len(avg_cost)), ysmoothed, color='blue', label= 'Sent every 200 time slots')
    plt.ylabel('Cost')
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":

    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 200
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY
    delay=[0]*NUM_IOT
    observation = []
    tasks= [0]*NUM_IOT
    prob=[0]*NUM_IOT
    cost=[0]*NUM_IOT

    # GENERATE ENVIRONMENT
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    iot_RL_list = list()
    lrates = [0.1,0.01,0.001,0.0001]
    
    
    #for lr in range(len(lrates)):
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                        learning_rate=0.0001,
                                        reward_decay=0.9,
                                        e_greedy=0.99,
                                        replace_target_iter=200,  # each 200 steps, update target net
                                        memory_size=500,  # maximum of memory
                                        batch_size=16
                                        ))

    # TRAIN THE SYSTEM
    train(iot_RL_list, NUM_EPISODE)
        

    for iot in range(len(cost)):
        cost[iot] = iot_RL_list[iot].cost_his
        #iot_RL_list[iot].plot()
        #iot_RL_list[iot].plot_loss()
        #iot_RL_list[iot].plot_delay()
    avg_cost = [float(sum(col))/len(col) for col in zip(*cost)] 
    print(avg_cost)
    plot_cost(avg_cost)
    
    
    
        #plot_delays(delay, prob)
    if rdt_vs_tap:
        # Export Ratio of Dropped Task vs Task Arrival Probability Info
        file_path = "RDT_vs_TAP.csv"
        if(os.path.exists(file_path)):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=['RDT', 'TAP'])
        df.loc[len(df.index)] = [sum(dropped_task_ratio)/NUM_EPISODE, env.task_arrive_prob]
        print(df)
        df.to_csv(file_path, index=False)
        
    if rdt_vs_nmd:
        # Export Ratio of Dropped Task vs Number of Mobile Devices
        file_path = "RDT_vs_NMD.csv"
        if(os.path.exists(file_path)):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=['RDT', 'NMD'])
        df.loc[len(df.index)] = [sum(dropped_task_ratio)/NUM_EPISODE, NUM_IOT]
        print(df)
        df.to_csv(file_path, index=False)
        
    if rdt_vs_td:
        # Export Ratio of Dropped Task vs Task Arrival Probability Info
        file_path = "RDT_vs_TD.csv"
        if(os.path.exists(file_path)):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=['RDT', 'TD'])
        df.loc[len(df.index)] = [sum(dropped_task_ratio)/NUM_EPISODE, env.duration]
        print(df)
        df.to_csv(file_path, index=False)
    
    
    #from scipy.ndimage.filters import gaussian_filter1d

    #ysmoothed = gaussian_filter1d(tasks.sort(reverse=True), sigma=2)
        #print(self.loss_his)

    
    
    #drop=[0]*len(tasks)
    
   
    #print(Offload.process_delay_unfinish_ind)
        
        
    print('Training Finished')
