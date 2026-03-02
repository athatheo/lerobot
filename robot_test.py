
# Follower /dev/tty.usbmodem5A7A0544481
# from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from argparse import ArgumentParser

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Tape Pickup"
FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0550051"
FOLLOWER_ID = "follower"
LEADER_PORT = "/dev/tty.usbmodem5A7A0544481"
LEADER_ID = "leader"
# CAMERA_CONFIG = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}

config = SO101FollowerConfig(
    port=FOLLOWER_PORT,
    id=FOLLOWER_ID,
    # cameras=CAMERA_CONFIG,
)
follower = SO101Follower(config)


# Leader /dev/tty.usbmodem5A7A0550051
config = SO101LeaderConfig(
    port=LEADER_PORT,
    id=LEADER_ID,
)
leader = SO101Leader(config)

def record():
    # Configure the dataset features
    action_features = hw_to_dataset_features(follower.action_features, "action")
    obs_features = hw_to_dataset_features(follower.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id="athatheo/so101-tape-pickup-test",
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    # Connect the robot and teleoperator
    follower.connect()
    leader.connect()

    # Create the required processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    episode_idx = 0
    while episode_idx < NUM_EPISODES or events["rerecord_episode"]:
        print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
        print("Starting record loop")
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=leader,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )
        print("Record loop finished")
        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            print("Starting reset loop")
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=leader,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            print("Reset loop finished")
        print("Saving episode")
        dataset.save_episode()
        print("Episode saved")
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    follower.disconnect()
    leader.disconnect()


def evaluate():
    """Run policy inference and record evaluation episodes."""
    # Initialize the policy from local pretrained model
    policy = ACTPolicy.from_pretrained(MODEL_PATH)

    # Configure the dataset features
    action_features = hw_to_dataset_features(follower.action_features, "action")
    obs_features = hw_to_dataset_features(follower.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create the evaluation dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_EVAL_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="evaluation")

    # Connect the robot
    follower.connect()

    # Create robot processors (required by record_loop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Create pre/post processors for policy inference
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_PATH,
        dataset_stats=dataset.meta.stats,
    )

    for episode_idx in range(EVAL_EPISODES):
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {EVAL_EPISODES}")

        # Run the policy inference loop
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EVAL_EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        dataset.save_episode()

        if events["stop_recording"]:
            break

    # Clean up
    log_say("Stop evaluation")
    follower.disconnect()


def teleop(follower, leader):
    follower.connect()
    leader.connect()
    while True:
        action = leader.get_action()
        follower.send_action(action)

def main():
    parser = ArgumentParser()
    parser.add_argument("--follower", action="store_true")
    parser.add_argument("--leader", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--teleop", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run policy inference/evaluation")
    args = parser.parse_args()

    if args.follower and not args.calibrate:
        follower.setup_motors()
    if args.leader and not args.calibrate:
        leader.setup_motors()

    if args.follower and args.calibrate:
        follower.connect(calibrate=False)
        follower.calibrate()
        follower.disconnect()
    if args.leader and args.calibrate:
        leader.connect(calibrate=False)
        leader.calibrate()
        leader.disconnect()
    
    if args.teleop:
        teleop(follower, leader)

    if args.record:
        record()

    if args.eval:
        evaluate()

if __name__ == "__main__":
    main()