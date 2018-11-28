import time
import traceback

import threading as th
import numpy as np
import tensorflow as tf

import gym
gym.logger.set_level(40)

from noise import one_fsq_noise

from history import History
from memory import Memory
from hyper_params import hyper


def w_init(stddev, inputs):
    return tf.truncated_normal_initializer(stddev=np.sqrt(stddev/inputs))


class Agent(object):
    def __init__(self,
                 observation_space_dims,
                 action_space,
                 discount_factor=.96,
                 model_path='./model'
                 ):

        self.discount_factor = discount_factor
        self.model_path = model_path
        self.global_step = 0
        self.history = History(log_path=model_path)
        self.max_reward = 1000
        self.lock = th.Lock()
        self.lock_swap = th.Lock()

        self.action_shape = action_space.shape               #(19,)
        self.observation_shape = (observation_space_dims, )  #(321,)
        self.inputdims = observation_space_dims
        self.memory = None
        self.block_training = False

        print("observation shape:", self.observation_shape)
        print("action shape: ", self.action_shape)

        self.is_continuous = True if isinstance(action_space, gym.spaces.Box) else False
        if self.is_continuous:
            low = action_space.low
            high = action_space.high
            num_of_actions = action_space.shape[0]
            
            self.action_bias = high/2. + low/2.
            self.action_multiplier = high - self.action_bias
            
            def clamp_action(actions):
                return np.clip(actions, a_max=action_space.high, a_min=action_space.low)

            self.clamp_action = clamp_action
        else:
            # not supported
            raise RuntimeError('This version of DDPG only supports continuous action space')

        self.outputdims = num_of_actions

        ids, ods = self.inputdims, self.outputdims
        #print('inputs:{}, outputs:{}'.format(ids, ods))

        # start TF
        #tf.reset_default_graph()
        self.tf_graph = tf.Graph()
        self.sess = tf.Session(graph=self.tf_graph)

        # setup model
        with self.tf_graph.as_default():
            self.nr_networks = hyper.nr_agents
            self.actor = []
            self.critic = []
            self.actor_target = []
            self.critic_target = []
            for i in range(self.nr_networks):
                self.actor.append( self.create_actor_network(ids, ods, 'actor_o'+str(i)) )
                self.critic.append( self.create_critic_network(ids, ods, 'critic_o'+str(i)) )
                self.actor_target.append( self.create_actor_network(ids, ods, 'actor_t'+str(i)) )
                self.critic_target.append( self.create_critic_network(ids, ods, 'critic_t'+str(i)) )

            # setup tf actions
            self.train, self.predict, self.sync_target, self.evaluate, self.swap_actors = self.train_step_gen()

            # setup model saving
            self.saver = tf.train.Saver(max_to_keep=10000)

            # init tf
            self.sess.run(tf.global_variables_initializer())
            # sync model => model_target (on first run)
            for i in range(self.nr_networks):
                self.sync_target(i)


    def setup_memory(self):
        if self.memory == None:
            print("Creating memory buffer, hold on...")
            limit = hyper.memory_size
            self.memory = Memory(limit=limit, action_shape=self.action_shape, observation_shape=self.observation_shape)


    def create_actor_network(self, num_inputs, num_outputs, scope):
        def actor_model(state):
            
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

                x = tf.layers.dense(state, 512, kernel_initializer=w_init(3.0, num_inputs), name='a1', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 512), name='a2', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 256), name='a3', reuse=tf.AUTO_REUSE)
                x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.leaky_relu(x, alpha=0.35)

                x = tf.layers.dense(x, 256, kernel_initializer=w_init(2.0, 256), name='a4', reuse=tf.AUTO_REUSE)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(2.0, 256), name='a5', reuse=tf.AUTO_REUSE)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(2.0, 256), name='a6', reuse=tf.AUTO_REUSE)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, num_outputs, kernel_initializer=w_init(0.5, 256), name='a9', reuse=tf.AUTO_REUSE)

                x = tf.nn.tanh(x) * self.action_multiplier + self.action_bias
                return x
        return actor_model


    def create_critic_network(self, num_inputs, num_outputs, scope):
        def critic_model(input):
            state = input[0]
            action = input[1]

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                x = tf.layers.dense(state, 256, kernel_initializer=w_init(3.0, num_inputs), name='c1s', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 256), name='c2s', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)

                y = tf.layers.dense(action, 256, kernel_initializer=w_init(3.0, num_outputs), name='c1a', reuse=tf.AUTO_REUSE)
                y = tf.nn.leaky_relu(y, alpha=0.35)

                x = tf.concat([x, y], axis=1)

                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 256+256), name='c2', reuse=tf.AUTO_REUSE)
                x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.leaky_relu(x, alpha=0.35)

                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 256), name='c3', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)
                x = tf.layers.dense(x, 256, kernel_initializer=w_init(3.0, 256), name='c4', reuse=tf.AUTO_REUSE)
                x = tf.nn.leaky_relu(x, alpha=0.35)

                x = tf.layers.dense(x, 1, kernel_initializer=w_init(1.0, 256), name='c9', reuse=tf.AUTO_REUSE)

                return x
        return critic_model


    def train_step_gen(self):
        s1 = tf.placeholder(tf.float32, shape=[None, self.inputdims])
        a1 = tf.placeholder(tf.float32, shape=[None, self.outputdims])
        r1 = tf.placeholder(tf.float32, shape=[None, 1])
        isdone = tf.placeholder(tf.float32, shape=[None, 1])
        s2 = tf.placeholder(tf.float32, shape=[None, self.inputdims])

        tau = tf.Variable(1e-3, name='tau', trainable=False)

        self.train_ops = []
        self.predict_ops = []
        self.sync_target_ops = []
        self.evaluate_ops = []
        self.actor_vars = []

        for i in range(self.nr_networks):
            scope = 'ac_'+str(i)
            with tf.variable_scope(scope):
                # 1. update the critic
                a2 = self.actor_target[i](s2)
                q2 = self.critic_target[i]([s2, a2])
                q1_target = r1 + (1-isdone) * (self.discount_factor + i*hyper.discount_step) * q2
                q1_predict = self.critic[i]([s1, a1])
                critic_loss = tf.reduce_mean((q1_target - q1_predict)**2)

                # 2. update the actor
                a1_predict = self.actor[i](s1)
                q1_predict2 = self.critic[i]([s1, a1_predict])
                actor_loss = tf.reduce_mean(- q1_predict2)

                # 3. shift the weights (aka target network)
                aw = tf.trainable_variables(scope=scope+'/actor_o')
                cw = tf.trainable_variables(scope=scope+'/critic_o')
                atw = tf.trainable_variables(scope=scope+'/actor_t')
                ctw = tf.trainable_variables(scope=scope+'/critic_t')
                self.actor_vars.append([aw, atw])
                one_m_tau = 1-tau
                shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(one_m_tau)) for i,_ in enumerate(aw)]
                shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(one_m_tau)) for i,_ in enumerate(cw)]

                # 4. inference
                a_infer = self.actor[i](s1)
                q_infer = self.critic[i]([s1, a_infer])

                # optimizer
                with tf.variable_scope('opt_a'):
                    opt_actor = tf.train.AdamOptimizer(hyper.lr_actor) #, name='Adam' default
                    astep = opt_actor.minimize(actor_loss, var_list=aw)
                with tf.variable_scope('opt_c'):
                    opt_critic = tf.train.AdamOptimizer(hyper.lr_critic) #, name='Adam'
                    cstep = opt_critic.minimize(critic_loss, var_list=cw)

                self.train_ops.append([critic_loss, actor_loss, cstep, astep, shift1, shift2])
                self.predict_ops.append([a_infer, q_infer])
                self.sync_target_ops.append([shift1, shift2])
                self.evaluate_ops.append([q1_predict])

        # setup ops for swapping actors
        self.copy_ops = []
        with tf.variable_scope('copy'):
            self.actor_backup = []
            self.actor_t_backup = []
            # Create variables of actor shape to hold a backup
            # I'm sure there's a better way to do this
            for _,av in enumerate(self.actor_vars[0][0]):
                self.actor_backup.append( tf.Variable(av) )
            for _,av in enumerate(self.actor_vars[0][1]):
                self.actor_t_backup.append( tf.Variable(av) )

            # copy the first one
            self.backup_cp = [tf.assign(self.actor_backup[k], self.actor_vars[0][0][k]) for k,_ in enumerate(self.actor_vars[0][0])]
            self.backup_cp_t = [tf.assign(self.actor_t_backup[k], self.actor_vars[0][1][k]) for k,_ in enumerate(self.actor_vars[0][1])]

            # copy actors to index-1
            for i in range(self.nr_networks-1):
                cp = [tf.assign(self.actor_vars[i][0][k], self.actor_vars[i+1][0][k]) for k,_ in enumerate(self.actor_vars[i][0])]
                cp_t = [tf.assign(self.actor_vars[i][1][k], self.actor_vars[i+1][1][k]) for k,_ in enumerate(self.actor_vars[i][1])]
                self.copy_ops.append([cp, cp_t])

            # copy the backup to the last
            last_id = self.nr_networks-1
            self.last_cp = [tf.assign(self.actor_vars[last_id][0][k], self.actor_backup[k]) for k,_ in enumerate(self.actor_vars[last_id][0])]
            self.last_cp_t = [tf.assign(self.actor_vars[last_id][1][k], self.actor_t_backup[k]) for k,_ in enumerate(self.actor_vars[last_id][1])]


        def swap_actors():
            with self.lock_swap:
                with self.lock:
                    # could setup control_dependencies/groups, but this is not run very often
                    self.sess.run([self.backup_cp, self.backup_cp_t], feed_dict={})
                    for _,cp in enumerate(self.copy_ops):
                        self.sess.run(cp, feed_dict={})
                    self.sess.run([self.last_cp, self.last_cp_t], feed_dict={})

        def train(memory, i):
            [s1d, a1d, r1d, isdoned, s2d] = memory
            res = self.sess.run(self.train_ops[i],
                                feed_dict={s1:s1d, a1:a1d, r1:r1d, isdone:isdoned, s2:s2d, tau:hyper.tau}
                                )
            return res

        def predict(state, i):
            res = self.sess.run(self.predict_ops[i], feed_dict={s1:state})
            return res

        def sync_target(i):
            self.sess.run(self.sync_target_ops[i], feed_dict={tau:1.})

        def evaluate(state, action, i):
            [qv] = self.sess.run(self.evaluate_ops[i], feed_dict={s1:state, a1:action})
            return qv

        return train, predict, sync_target, evaluate, swap_actors


    def test_swap_actors(self):
        for i in range(self.nr_networks):
            print(self.sess.run(self.actor_vars[i][0][0][0][0], feed_dict={}))


    def get_max_action(self, observation):
        obs_b = np.reshape(observation, (1,len(observation)))

        # get actions
        all_actions = []
        for aci in range(self.nr_networks):
            [actions, _] = self.predict(obs_b, aci)
            # setup for batches, get first
            action = actions[0]
            all_actions.append(action)

        # create combinations
        for ai in range(self.nr_networks):
            for aj in range(self.nr_networks):
                if aj<ai:
                    a1 = all_actions[ai]
                    a2 = all_actions[aj]
                    avg_action = (a1 + a2) * 0.5
                    all_actions.append(avg_action)

        # and more combinations
        for ai in range(self.nr_networks):
            for aj in range(self.nr_networks):
                for ak in range(self.nr_networks):
                    if aj<ai and ak<aj:
                        a1 = all_actions[ai]
                        a2 = all_actions[aj]
                        a3 = all_actions[ak]
                        avg_action = (a1 + a2 + a3) * (1.0/3.0)
                        all_actions.append(avg_action)

        # for sanity
        for ai in range(len(all_actions)):
            all_actions[ai] = self.clamp_action(all_actions[ai])

        # make it a np array, so we can batch it
        all_actions = np.asarray(all_actions)
        # stack observation for batching
        all_obs = np.repeat(obs_b, len(all_actions), axis=0)
        # get qv from each critic and sum them
        for ci in range(self.nr_networks):
            if ci==0:
                all_qv_b = self.evaluate(all_obs, all_actions, ci)
            else:
                all_qv_b += self.evaluate(all_obs, all_actions, ci)
        all_qv_b /= self.nr_networks

        # evaluate actions
        max_action = None
        max_qv = None
        for ai in range(len(all_qv_b)):
            qv_a = all_qv_b[ai]
            if max_qv==None or qv_a > max_qv:
                max_action = all_actions[ai]
                max_qv = qv_a

        return max_action


    def get_all_actions(self, observation):
        obs_b = np.reshape(observation, (1,len(observation)))
        # get actions
        all_actions = []
        for aci in range(self.nr_networks):
            [actions, _] = self.predict(obs_b, aci)
            # setup for batches, get first
            action = actions[0]
            all_actions.append(action)
        return all_actions

    def get_action_qs(self, observation, all_actions):
        obs_b = np.reshape(observation, (1,len(observation)))
        # make it a np array, so we can batch it
        all_actions = np.asarray(all_actions)
        # stack observation for batching
        all_obs = np.repeat(obs_b, len(all_actions), axis=0)
        # get qv from each critic and sum them
        for ci in range(self.nr_networks):
            if ci==0:
                all_qv_b = self.evaluate(all_obs, all_actions, ci)
            else:
                all_qv_b += self.evaluate(all_obs, all_actions, ci)
        all_qv_b /= self.nr_networks
        return all_qv_b


    def get_action(self, observation, i):
        obs = np.reshape(observation, (1,len(observation)))
        [actions, q] = self.predict(obs, i)
        actions, q = actions[0], q[0]
        return actions


    def train_batch(self, i):
        if self.block_training:
            return
        # only if enough samples in memory
        if self.memory.size() > hyper.batch_size * 128:
            # sample a minibatch
            [s1, a1, r1, isdone, s2] = self.memory.sample_batch(hyper.batch_size)
            # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)
            self.train([s1, a1, r1, isdone, s2], i)

    def append_memory(self, s1, a1, r1, isdone, s2):
        self.memory.append(s1, a1, r1, isdone, s2)


    def run_episode(self, fenv, max_steps=-1, training=False, render=False, noise_level=0., ac_id=0):
        time_start = time.time()

        noise_source = None
        if noise_level > 0.0:
            noise_source = one_fsq_noise()
            # warm up noise source
            for _ in range(2000):
                noise_source.one((self.outputdims,), noise_level)

        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0

        try:
            # this might be a remote env
            observation = np.array( fenv.reset() )
        except Exception as e:
            print('Bad things during reset. Episode terminated.', e)
            traceback.print_exc()
            return

        while True and steps <= max_steps:
            steps +=1

            observation_before_action = observation # s1

            exploration_noise = 0.0
            if noise_level > 0.0:
                exploration_noise = noise_source.one((self.outputdims,), noise_level)

            # get action
            action = None
            with self.lock_swap:
                if training:
                    action = self.get_action(observation_before_action, ac_id)
                else:
                    action = self.get_max_action(observation_before_action)

            # add noise to our actions, since our policy is deterministic
            if noise_level > 0.0:
                exploration_noise *= self.action_multiplier
                action += exploration_noise
            action = self.clamp_action(action)

            # step
            try:
                # can't send receive np arrays over pyro
                action_out = [float(action[i]) for i in range(len(action))]
                observation, reward, done, _info = fenv.step(action_out)
                observation = np.array( observation )
            except Exception as e:
                print('Bad things during step. Episode terminated.', e)
                traceback.print_exc()
                return

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            # train
            if training == True:
                # The code works without this lock, but depending on training speed there is too much noise on updates.
                # The model always trains and is more stable with lock here
                with self.lock:
                    self.append_memory(observation_before_action, action, reward, isdone, observation)    # s1,a1,r1,isdone,s2
                    for i in range(self.nr_networks):
                        self.train_batch(i)
            else:
                if render:
                    fenv.render()

            if done:
                break

        totaltime = time.time() - time_start

        if training == True:
            self.global_step += 1
            print(self.global_step, ': Episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format( steps,totaltime,totaltime/steps,total_reward ))
            self.history.append_train( total_reward, noise_level, steps )
        else:
            print('Test done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format( steps,totaltime,totaltime/steps,total_reward ))
            self.history.append_test( total_reward, self.global_step, steps )
            if render==False:
                # background test
                if total_reward > self.max_reward:
                    self.max_reward = total_reward
                    self.save_weights("max_model")
                    print("Saved new max model with score: ", total_reward)

        return total_reward


    def save_weights(self, model_name="model"):
        with self.lock_swap:
            with self.lock:
                self.saver.save(self.sess, self.model_path + "/" + model_name, global_step = self.global_step)
        print("Saved model at global episode:", self.global_step)

    def load_weights(self, model=""):
        print('Loading Model...')
        path = ""
        if model == "":
            checkpoint = tf.train.get_checkpoint_state(self.model_path)
            if checkpoint:
                path = checkpoint.model_checkpoint_path
        else:
            path = self.model_path + "/" + model
        try:
            self.saver.restore(self.sess, path)
            print("Loaded model from checkpoint:", path)
            return True
        except Exception as ex:
            print("No model checkpoint available!")
            return False
