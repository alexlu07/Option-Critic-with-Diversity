import gym
import torch
from config import Config

from train import Trainer

def train(env, save=True, load=True, epoch=None, save_interval=10, n_envs=4, asynchronous = False):
    config = Config()
    config.make_env(env, n_envs, asynchronous=asynchronous)

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

        if save and epoch % save_interval == 0:
            trainer.save_state(save_interval)
            with open("./results/log.txt", "a") as f:
                f.write(log + "\n")

def watch(env, epoch):
    config = Config()
    config.make_env(env, 1,render_mode="human")

    trainer = Trainer(config)
    trainer.load_state(epoch)

    changed = True

    obs = trainer.obs
    opt = trainer.opt

    for i in range(2048):
        trainer.env.call("render")

        if changed:
            print("======== OPTION", opt, "========")

        changed = False

        act, logp, next_opt, optval, val, termprob = trainer.model.step(obs, opt)
        next_obs, rew, done, terminated, info = trainer.env.step(act)

        if next_opt != opt:
            changed = True

        opt = next_opt

        obs = torch.as_tensor(next_obs, dtype=torch.float32)
        
        if done:
            opt[done] = trainer.model.get_option_dist(trainer.model.get_state(obs[done]))[0]
            changed = True

    trainer.env.close()
    


# watch("CartPole-v1", 700)
train("CartPole-v1", load=False, save=False, asynchronous=False)