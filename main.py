import csv
import gymnasium as gym
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from collections import deque
from config import Config
import time

from train import Trainer

def train(env, pretrain=False, save=True, load=True, epoch=None, save_interval=100, n_envs=1, asynchronous = False):

    # table = {name: 0 for name in ["ep_len", "ep_ret", "term_probs", "disc_loss", "opt_usage", "opt_probs", "ep_rets"]}
    # for key in table:
    #     with open("data/" + key + ".csv", 'r') as f:
    #         csvreader = csv.reader(f)
    #         table[key] = list(csvreader)


    table = {name: [[0 for a in range(3)] for b in range(701)] for name in ["ep_len", "ep_ret", "term_probs", "disc_loss"]}
    table["opt_usage"] = [[0 for a in range(3 * 2)] for b in range(701)]
    table["opt_probs"] = [[0 for a in range(3 * 2)] for b in range(701)]
    table["ep_rets"] = []

    for a in range(2, 3):
        config = Config()
        config.make_env(env, n_envs, asynchronous=asynchronous)

        trainer = Trainer(config)
        writer = SummaryWriter("results/tb_log")

        if load:
            if epoch:
                trainer.load_state(epoch)
            else:
                trainer.load_state("current_checkpoint")

        if pretrain:
            epoch = 0
            while epoch < 300:
                epoch, pi_loss, vf_loss, term_loss, disc_loss, ep_len, ep_ret, opt_usage, opt_probs, termprobs, rollout_time, training_time, eps = trainer.train_one_epoch(pretrain=True)


            trainer.epoch = 0

        rets_deque = deque(maxlen=150)
        rets_counter = 0
        epoch = 0
        while epoch < 700:
            epoch, pi_loss, vf_loss, term_loss, disc_loss, ep_len, ep_ret, opt_usage, opt_probs, termprobs, rollout_time, training_time, eps = trainer.train_one_epoch()
            for i in ep_ret:
                rets_deque.append(i)
                writer.add_scalar("ep_rets", sum(rets_deque) / len(rets_deque), rets_counter)
                if len(table["ep_rets"]) <= rets_counter:
                    table["ep_rets"].append([0 for x in range(3)])
                table["ep_rets"][rets_counter][a] = sum(rets_deque) / len(rets_deque)
                rets_counter += 1

            ep_len = sum(ep_len)/len(ep_len)
            ep_ret = sum(ep_ret)/len(ep_ret)
            

            opt_usage = np.array(opt_usage)
            opt_probs = np.array(opt_probs)
            opt_usage = opt_usage.sum(axis=0)/opt_usage.sum()
            opt_probs = opt_probs.sum(axis=0)/opt_probs.sum()
            ep_opt = [f"{i:.4f}" for i in opt_usage]

            log = f"{epoch}: (act_loss: {pi_loss:.4f}, crit_loss: {vf_loss:.8f}, term_loss: {term_loss:.4f}, ep_len: {ep_len:.4f}, ep_ret: {ep_ret:.4f}, opt_usage: [{', '.join(ep_opt)}], eps: {eps}, ep_time: {rollout_time:.4f}, train_time: {training_time:.4f})"

            print(log)

            writer.add_scalar("ep_len", ep_len, epoch)
            writer.add_scalar("ep_ret", ep_ret, epoch)
            writer.add_scalar("pi_loss", pi_loss, epoch)
            writer.add_scalar("vf_loss", vf_loss, epoch)
            writer.add_scalar("term_loss", term_loss, epoch)
            writer.add_scalar("term_probs", termprobs.mean(), epoch)
            writer.add_scalar("term_std", termprobs.std(), epoch)
            writer.add_scalar("disc_loss", disc_loss, epoch)

            table["ep_len"][epoch][a] = ep_len
            table["ep_ret"][epoch][a] = ep_ret
            table["term_probs"][epoch][a] = termprobs.mean()
            table["disc_loss"][epoch][a] = disc_loss

            opt_usage = np.cumsum(opt_usage)
            opt_probs = np.cumsum(opt_probs)
            writer.add_scalars("opt_usage", {str(i): x for i, x in enumerate(opt_usage)}, epoch)
            writer.add_scalars("opt_probs", {str(i): x for i, x in enumerate(opt_probs)}, epoch)

            for i, x in enumerate(opt_usage):
                table["opt_usage"][epoch][a*2 + i] = x

            for i, x in enumerate(opt_probs):
                table["opt_probs"][epoch][a*2 + i] = x                

            for key in table:
                with open("data/" + key + ".csv", 'w') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerows(table[key])

            if save and epoch % save_interval == 0:
                trainer.save_state(save_interval)

def watch(env, epoch, force_opt=None):
    config = Config()
    config.testing = True
    config.make_env(env, 1, render_mode="human")

    trainer = Trainer(config)
    trainer.load_state(epoch)

    changed = True

    obs = trainer.obs
    opt = trainer.opt

    time.sleep(3)

    for i in range(2048):
        trainer.env.call("render")

        if changed:
            print("======== OPTION", opt, "========")

        changed = False

        act, logp, next_opt, optval, val, termprob, term = trainer.model.step(obs, opt, i, force_opt=force_opt)
        next_obs, rew, done, terminated, info = trainer.env.step(act)

        # print(optdist, no, next_opt, val*1000)
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
    


watch("CartPole-v1", "600")
# train("MiniGrid-FourRooms-v0", n_envs=1, load=False, save=False, asynchronous=False)
# train("fourrooms", n_envs=1, load=False, save=False, asynchronous=False)
# train("CartPole-v0", n_envs=1, load=False, save=True, asynchronous=False)