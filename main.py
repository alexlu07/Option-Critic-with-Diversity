from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from config import Config

from train import Trainer

env = make_vec_env("MountainCar-v0", n_envs=4, seed=0, vec_env_cls=DummyVecEnv)

def train(env, save=True, load=True, epoch=None, save_interval=10):
    config = Config(env)
    trainer = Trainer(config)

    if load:
        if epoch:
            trainer.load_state(epoch)
        else:
            trainer.load_state("current_checkpoint")

    while True:
        epoch, ep_len, ep_ret, rollout_time, training_time = trainer.train_one_epoch()
        ep_len = sum(ep_len)/len(ep_len)
        ep_ret = sum(ep_ret)/len(ep_ret)


        log = f"{epoch}: (ep_len: {ep_len}, ep_ret: {ep_ret}, ep_time: {rollout_time:.4f}, train_time: {training_time:.4f})"
        print(log)

        if save:
            trainer.save_state(save_interval)
            with open("./results/log.txt", "a") as f:
                f.write(log + "\n")

train(env, load=False)