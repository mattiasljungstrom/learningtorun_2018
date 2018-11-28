import json

# Training history, saving to disk and visualization

class History(object):    
    def __init__(self, log_path):
        self.first_view = True
        self.log_path = log_path
        self.train_rewards = []
        self.train_noise = []
        self.train_steps = []
        self.test_rewards = []
        self.test_episode = []
        self.test_steps = []
        self.dirty = False
        self.plot_disabled = False

    def append_train(self, reward, noise, steps):
        self.train_rewards.append(reward)
        self.train_noise.append(noise)
        self.train_steps.append(steps)
        self.dirty = True

    def append_test(self, reward, episode, steps):
        self.test_rewards.append(reward)
        self.test_episode.append(episode)
        self.test_steps.append(steps)
        self.dirty = True

    def get_train_count(self):
        return len(self.train_rewards)

    def get_last_noise_level(self, noiseless):
        i = len(self.train_noise) - 1
        # find last noise that wasn't noiseless
        # not best code ever, assumes noiseless != noise_floor
        while i > 0:
            noise = self.train_noise[i]
            if noise!=noiseless:
                return noise
            i -= 1
        return 2.0

    def print_last_tests(self, last_nr):
        last_index = len(self.test_rewards) - 1
        while last_index >= 0 and last_nr >= 0:
            print("test:", self.test_episode[last_index], ":", self.test_rewards[last_index], ":", self.test_steps[last_index])
            last_nr -= 1
            last_index -= 1

    def plot(self):
        if not self.dirty or self.plot_disabled:
            return
        self.dirty = False

        if self.first_view:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                self.plot_disabled = True
                return

            self.first_view = False
            plt.ion()
            self.fig = plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
            self.plot1 = plt.subplot(2,1,1) # nrows, ncols, index
            self.plot2 = plt.subplot(2,1,2)

        self.plot1.clear()
        self.plot1.grid(True)
        #self.plot1.ylabel('reward')
        self.plot1.set_ylim([-2000,10000])
        self.line_train, = self.plot1.plot(self.train_rewards)
        #x_values_test = np.linspace(0, len(self.train_rewards), len(self.test_rewards))
        #self.line_test, = self.plot1.plot(x_values_test, self.test_rewards)
        self.line_test, = self.plot1.plot(self.test_episode, self.test_rewards)    
        self.plot1.legend([self.line_train, self.line_test], ['train reward', 'test reward'])

        self.plot2.clear()
        self.plot2.grid(True)
        self.plot2.set_ylim([0,3])
        self.line_noise, = self.plot2.plot(self.train_noise)
        self.plot2.legend([self.line_noise], ['noise'])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def plot_train_test(self):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        # create more sensible x-axis from nr of steps
        # (since early episodes will be much shorter)
        train_experiences = np.cumsum(self.train_steps)
        test_experiences = [train_experiences[i] for i in self.test_episode]

        self.fig = plt.figure(figsize=(7, 4), dpi=300, facecolor='w', edgecolor='k')
        self.plot1 = plt.subplot(1,1,1) # nrows, ncols, index

        self.plot1.clear()
        self.plot1.grid(True)
        #self.plot1.ylabel('reward')
        self.plot1.set_ylim([-2000,10000])
        #self.plot1.set_xlim([0,23000])
        self.plot1.set_xlim([0,2.3e6])
        self.plot1.plot(train_experiences, self.train_rewards, label='train rewards')
        #self.plot1.plot(self.test_episode, self.test_rewards, label='test reward')
        self.plot1.plot(test_experiences, self.test_rewards, label='test reward')
        self.plot1.legend(loc='upper left')

        self.plot1.set_xlabel('training steps')

        #self.fig.tight_layout()
        plt.show()


    def save(self):
        path = self.log_path + '/history_data.log'
        print('Writing history data.')
        with open(path, 'wt') as out_file:
            json.dump([self.train_rewards, self.train_noise, 
                       self.test_rewards, self.test_episode, 
                       self.train_steps, self.test_steps], 
                       out_file, sort_keys=False, indent=0, separators=(',', ':'))

    def load(self):
        path = self.log_path + '/history_data.log'
        print('Loading history data...')
        try:
            with open(path, 'r') as in_file:
                input_data = json.load(in_file)
                self.train_rewards = input_data[0]
                self.train_noise = input_data[1]
                self.test_rewards = input_data[2]
                self.test_episode = input_data[3]
                if len(input_data) > 4:
                    self.train_steps = input_data[4]
                    self.test_steps = input_data[5]
            self.dirty = True
        except Exception as ex:
            print('No history data found.')



if __name__=='__main__':

    model_path = './model_prosthetics_v2'
    history = History(model_path)
    history.load()
    history.plot_train_test()

