import gym
import torch
import numpy as np
from config import Config
import time

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
        epoch, pi_loss, vf_loss, term_loss, ep_len, ep_ret, ep_opt, rollout_time, training_time, eps = trainer.train_one_epoch()
        ep_len = sum(ep_len)/len(ep_len)
        ep_ret = sum(ep_ret)/len(ep_ret)

        ep_opt = np.array(ep_opt)
        ep_opt = ep_opt.sum(axis=0)/ep_opt.sum()
        ep_opt = [f"{i:.4f}" for i in ep_opt]

        log = f"{epoch}: (act_loss: {pi_loss:.4f}, crit_loss: {vf_loss:.8f}, term_loss: {term_loss:.4f}, ep_len: {ep_len:.4f}, ep_ret: {ep_ret:.4f}, opt_usage: [{', '.join(ep_opt)}], eps: {eps}, ep_time: {rollout_time:.4f}, train_time: {training_time:.4f})"
        print(log)

        if save and epoch % save_interval == 0:
            trainer.save_state(save_interval)
            with open("./results/log.txt", "a") as f:
                f.write(log + "\n")

def watch(env, epoch):
    config = Config()
    config.testing = True
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

        act, logp, next_opt, optval, val, termprob, term, optdist, no = trainer.model.step(obs, opt, i)
        next_obs, rew, done, terminated, info = trainer.env.step(act)

        print(optdist, no, next_opt, val*1000)
        # time.sleep(0.2)

        if term:
            print("term")

        if next_opt != opt:
            changed = True

        opt = next_opt

        obs = torch.as_tensor(next_obs, dtype=torch.float32)
        
        if done:
            opt[done] = trainer.model.get_option_dist(trainer.model.get_state(obs[done]))[0]
            changed = True

    trainer.env.close()
    


# watch("CartPole-v1", "current_checkpoint")
train("CartPole-v1", load=False, save=False, asynchronous=False)