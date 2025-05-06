import hydra
from dex.trainers.rl_trainer import RLTrainer

@hydra.main(version_base=None, config_path="./dex/configs", config_name="train")
def main(cfg):
    print(cfg)
    exp = RLTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()


# # @hydra.main(version_base=None, config_path="./dex/configs", config_name="train")
# # def main(cfg):
# #     print(cfg)
# #     {'agent': {'name': 'DEX', 'device': 'cuda', 'discount': 0.99, 'reward_scale': 1, 'n_seed_steps': 1000, 'actor_lr': 0.001, 'critic_lr': 0.001, 
# #            'noise_eps': 0.1, 'aux_weight': 5, 'p_dist': 2, 'k': 5, 'soft_target_tau': 0.005, 'clip_obs': 200, 'norm_clip': 5, 'norm_eps': 0.01,
# #              'hidden_dim': 256, 'sampler': {'type': 'her_seq', 'strategy': 'future', 'k': 4}, 'update_epoch': 40}, 
# #              'cwd': '/home/amrit/pipeline/in_progress/Surgery/DEX/exp_local', 
# #              'n_train_steps': 100001, 'n_eval': 10, 'n_save': 5, 'n_log': 100, 'num_demo': 200, 
# #              'n_seed_steps': 32,
# #                'replay_buffer_capacity': 100000, 'batch_size': 128, 'device': 'cuda:0', 'seed': 1, 'task': 'TissueRetract-v0', 'postfix': None, 
# #                'dont_save': False, 'n_eval_episodes': 20, 'use_wb': True, 'project_name': 'dex', 'entity_name': 'dr_amrit',
# #                  'mpi': {'rank': None, 'is_chef': None, 'num_workers': None}}

# #     exp = RLTrainer(cfg)
# #     exp.train()

# # if __name__ == "__main__":
# #     main()

# cfg = {'agent': {'name': 'DEX', 'device': 'cuda', 'discount': 0.99, 'reward_scale': 1, 'n_seed_steps': 1000, 'actor_lr': 0.001, 'critic_lr': 0.001, 
#         'noise_eps': 0.1, 'aux_weight': 5, 'p_dist': 2, 'k': 5, 'soft_target_tau': 0.005, 'clip_obs': 200, 'norm_clip': 5, 'norm_eps': 0.01,
#             'hidden_dim': 256, 'sampler': {'type': 'her_seq', 'strategy': 'future', 'k': 4}, 'update_epoch': 40}, 
#             'cwd': '/home/amrit/pipeline/in_progress/Surgery/DEX/exp_local', 
#             'n_train_steps': 100001, 'n_eval': 10, 'n_save': 5, 'n_log': 100, 'num_demo': 200, 
#             'n_seed_steps': 32,
#             'replay_buffer_capacity': 100000, 'batch_size': 128, 'device': 'cuda:0', 'seed': 1, 'task': 'TissueRetract-v0', 'postfix': None, 
#             'dont_save': False, 'n_eval_episodes': 20, 'use_wb': True, 'project_name': 'dex', 'entity_name': 'dr_amrit',
#                 'mpi': {'rank': None, 'is_chef': None, 'num_workers': None}
#                 }
# print(cfg.keys())

# exp = RLTrainer(cfg)
# exp.train()
