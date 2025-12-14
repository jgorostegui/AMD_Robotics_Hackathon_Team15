#!/usr/bin/env python3  

# CAMERA_CONFIG="{camera1: ${TOP_INDEX}, camera2: {${SIDE_INDEX}, camera3: {${GRIPPER_INDEX},"

# TOP_CAMERA=/dev/video4          # Logitech overhead camera
# SIDE_CAMERA=/dev/video2         # Side view camera
# GRIPPER_CAMERA=/dev/video6 

"""Script de inferencia sin guardar dataset, teclado ni rename_map."""  
  
import time  
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  
from lerobot.policies.factory import make_policy, make_pre_post_processors  
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig  
from lerobot.robots.so101_follower.so101_follower import SO101Follower  
from lerobot.scripts.lerobot_record import record_loop  
from lerobot.utils.utils import log_say  
from lerobot.utils.visualization_utils import init_rerun
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import make_default_processors
  
  
def main():  
    # Configuración - MODIFICA ESTOS VALORES  
    FPS = 30  
    EPISODE_TIME_SEC = 60  
    TASK_DESCRIPTION = "Pick the orange ball and place it in column 0." 
    POLICY_PATH = "outputs/train/mission2_smolvla_multitask_30ksteps_checkpoint10k/checkpoints/last/pretrained_model"
    DATASET_REPO_ID = "jlamperez/mission2_smolvla_multitask"
      
    # Configuración de cámaras (camera1, camera2, camera3 como espera SmolVLA)  
    camera_config = {  
        "camera1": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),  
        "camera2": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),  
        "camera3": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS)  
    }

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()  

      
    # Crear robot  
    robot_config = SO101FollowerConfig(  
        port="/dev/ttyACM1",  
        id="my_awesome_follower_arm",  
        cameras=camera_config  
    )  
    robot = SO101Follower(robot_config)

    dataset = LeRobotDataset(DATASET_REPO_ID)

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)  
    policy_cfg.device = "cuda" 
      
    # Cargar política  
    # policy_cfg = {  
    #     "pretrained_path": POLICY_PATH,  
    #     "device": "cuda"  
    # }  
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)  
      
    # Crear procesadores sin rename_map ni stats  
    preprocessor, postprocessor = make_pre_post_processors(  
        policy_cfg=policy.config,  
        pretrained_path=POLICY_PATH,  
        dataset_stats=dataset.meta.stats, 
        preprocessor_overrides={  
            "device_processor": {"device": "cuda"},  
        },  
    )  
      
    # Inicializar visualización (sin teclado)  
    init_rerun(session_name="inference")  
      
    # Crear diccionario de eventos vacío  
    events = {  
        "exit_early": False,  
        "rerecord_episode": False,  
        "stop_recording": False  
    }  
      
    # Conectar robot  
    robot.connect()  
      
    # Ejecutar inferencia sin guardar  
    log_say("Ejecutando inferencia sin guardar dataset")  
      
    record_loop(  
        robot=robot,  
        events=events,  
        fps=FPS,
        teleop_action_processor=teleop_action_processor,  # <-- ADD  
        robot_action_processor=robot_action_processor,    # <-- ADD  
        robot_observation_processor=robot_observation_processor,
        policy=policy,  
        preprocessor=preprocessor,  
        postprocessor=postprocessor,  
        dataset=dataset,  # No guardar datos  
        control_time_s=EPISODE_TIME_SEC,  
        single_task=TASK_DESCRIPTION,  
        display_data=True,  
    )  
      
    # Limpiar  
    robot.disconnect()  
    log_say("Inferencia completada")  
  
  
if __name__ == "__main__":  
    main()