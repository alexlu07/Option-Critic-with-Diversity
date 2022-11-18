import optuna

from config import Config
from train import Trainer


# minibatch_size, num_options, lr, gamma, lam, termination_reg
# model architecture
def objective(trial):
    minibatch_size = trial.suggest_int("minibatch_size", 128, 512, 128)
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    gamma = trial.suggest_float("gamma", 0.7, 1)
    lam = trial.suggest_float("lam", 0.6, 1)
    termination_reg = trial.suggest_float("termination_reg", 0, 0.4)

    feature_arch = [trial.suggest_int(f"feature_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("feature_arch_layers", 0, 3))]
    critic_arch = [trial.suggest_int(f"critic_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("critic_arch_layers", 0, 2))]
    term_arch = [trial.suggest_int(f"term_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("term_arch_layers", 0, 2))]
    opt_arch = [trial.suggest_int(f"opt_arch{i}", 64, 256, 64) for i in range(trial.suggest_int("opt_arch_layers", 0, 2))]
    num_options = trial.suggest_int("num_options", 2, 2)

    config = Config()
    config.make_env("CartPole-v1", 4, asynchronous=False)

    config.minibatch_size = minibatch_size
    config.lr = lr
    config.gamma = gamma
    config.lam = lam
    config.termination_reg = termination_reg

    config.feature_arch = feature_arch
    config.critic_arch = critic_arch
    config.term_arch = term_arch
    config.opt_arch = opt_arch
    config.num_options = num_options

    trainer = Trainer(config)

    for i in range(150):
        epoch, pi_loss, vf_loss, term_loss, ep_len, ep_ret, ep_opt, rollout_time, training_time, eps = trainer.train_one_epoch()

        ep_ret = sum(ep_ret)/len(ep_ret)
        print(ep_ret, i)

        trial.report(ep_ret, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return ep_ret

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)