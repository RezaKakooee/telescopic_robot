import rootutils
rootutils.setup_root('/home/azureuser/telescopic_robot', pythonpath=True)
from radial_sphere import MujocoSteeringEnv, load_config

cfg = load_config()
print('Testing baseline steering env creation...')
env = MujocoSteeringEnv(cfg)
obs, info = env.reset()
print(f'Baseline env initialized successfully! Obs shape: {obs.shape}, Sim2Real enabled: {env.enable_sim2real}')
env.close()

cfg.sim2real.enabled = True
cfg.sim2real.enable_actuator_limits = True
cfg.sim2real.enable_sensor_noise = True
cfg.sim2real.enable_latency = True
print('Testing Sim2Real steering env creation...')
env2 = MujocoSteeringEnv(cfg)
obs2, info2 = env2.reset()
print(f'Sim2Real env initialized successfully! Obs shape: {obs2.shape}, Sim2Real enabled: {env2.enable_sim2real}')
env2.close()
