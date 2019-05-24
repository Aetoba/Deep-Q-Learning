from collections import deque

import numpy as np

# from PIL import Image
from skimage import transform
from skimage import color

def preprocess_frame(frame):
    """
    Makes frame grayscale and correct size (105,80)
    """

    # img = Image.fromarray(frame, 'RGB')
    # img.save('/screenshots/my.png')

    frame = np.mean(frame, axis=2).astype(np.uint8) # make grayscale
    frame = frame[::2, ::2] # downsample to 105x80

    # img = Image.fromarray(frame, 'L')
    # img.save('./data/not_runs/before_after_processing/improved.png')
    return frame

def phi(new_screen, frame_stack_size, new_episode=False, curr_state=deque([])):
    """
    Takes new screen and current state (stack of past four frames) and makes new state
    """

    if new_episode:
        state = deque(maxlen=frame_stack_size)
        frame = preprocess_frame(new_screen)
        for _ in range(frame_stack_size):
            state.append(frame)
    else:
        state = curr_state
        state.append(preprocess_frame(new_screen))

    return state, np.stack(state, axis=2)
