import numpy as np
import random
from PIL import Image
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation
from keras.optimizers import Adam
sizes = (84, 84, 1)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=400000)
        self.gamma = 0.99   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.025  # exploration will not decay futher
        self.epsilon_decay = 0.00024375
        self.learning_rate = 0.0005
        self.loss = 0
        self.model = self._build_model()
        self.weight_backup = 'model_weights.h5'
        self.old_I_2 = None
        self.old_I_3 = None
        self.old_I_4 = None
        self.old_I_1 = None
        self.f = open('csvfile.csv', 'w')

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, activation='relu', strides=2, padding='same', input_shape=sizes))
        model.add(Conv2D(64, kernel_size=6, activation='relu', strides=2, padding='same'))
        model.add(Conv2D(64, kernel_size=3, activation='relu', strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size))
        #model.add(Activation('softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            print(act_values)
            return np.argmax(act_values[0])

    def RGBprocess(self, raw_img):
        processed_observation = Image.fromarray(raw_img, 'RGB')
        processed_observation = processed_observation.convert('L')
        processed_observation = processed_observation.resize((84, 84))
        processed_observation = np.array(processed_observation)
        processed_observation = processed_observation.reshape(
            1, processed_observation.shape[0], processed_observation.shape[1], 1)
        return processed_observation

    def stack(self, processed_observation, old_processed_observation):
        processed_stack = np.maximum(processed_observation, old_processed_observation)
        # I_4 = self.old_I_3 if self.old_I_3 is not None else np.zeros(
        #     (1, 84, 84))
        # I_3 = self.old_I_2 if self.old_I_2 is not None else np.zeros(
        #     (1, 84, 84))
        # I_2 = self.old_I_1 if self.old_I_1 is not None else np.zeros(
        #     (1, 84, 84))
        # I_1 = processed_observation
        # processed_stack = np.stack((I_4, I_3, I_2, I_1), axis=3)
        # self.old_I_4 = I_4
        # self.old_I_3 = I_3
        # self.old_I_2 = I_2
        # self.old_I_1 = I_1
        return processed_stack

    def remember(self, state, action, reward, new_state, done):
        if len(self.memory) >= 400000:
            self.memory.popleft()
            self.memory.append([state, action, reward, new_state, done])
        else:
            self.memory.append([state, action, reward, new_state, done])

    def memory_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        Sample = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in Sample:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(new_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
        self.f.write('{}, {}\n'.format(history.history['loss'][0], target_f[0][action]))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
