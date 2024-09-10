import torch

class Constants:
    NUMBER_OF_SUITS = 4
    NUMBER_OF_CARDS_PER_SUIT = 13
    NUMBER_OF_CARDS = NUMBER_OF_SUITS * NUMBER_OF_CARDS_PER_SUIT

    N_CARDS_TO_TEXT = {0:'a', 1: 'a pair of', 2: 'a triplet of', 3: 'a quad of', 4: 'PASS!'}

    NUMBER_OF_POSSIBLE_STATES = (NUMBER_OF_CARDS_PER_SUIT+1)*NUMBER_OF_SUITS + 1

    REPLAY_MEMORY_SIZE = 50_000 ## 
    MIN_REPLAY_MEMORY_SIZE = 1_000
    MODEL_NAME = "256x2"

    BATCH_SIZE = 256
    DISCOUNT = .99
    UPDATE_TARGET_EVERY = 1000

    EPISODES = 1000

    REWARD_PASS = -0.1
    REWARD_CARD = 0.1

    REWARD_WIN = 10
    REWARD_SECOND = 2
    REWARD_THIRD = 0
    REWARD_FOURTH = -5
    REWARD_LOSE = -10

    EPSILON = 1
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    AGGREGATE_STATS_EVERY = 10
