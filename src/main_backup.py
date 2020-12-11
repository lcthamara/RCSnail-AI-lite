import time
import logging
import traceback
import asyncio
import signal
import numpy as np
import zmq
import cv2
import copy
from zmq.asyncio import Context

import os

import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager

# from src.utilities.transformer import Transformer
from utilities.recorder import Recorder

car_mask = cv2.imread('car-mask-120x120.png',0)

def drive_backwards(frame, old_frame):
    dif = np.sum(np.absolute(np.array(frame) - np.array(old_frame)))
    print(dif)
    if (dif > 800):
        print("miau")
        return False
    else:
        return True

def process_frame(frame):
  # resize
  camera_dim = (160, 80) # (width, height)
  frame = cv2.resize(frame, camera_dim, interpolation = cv2.INTER_AREA)

  # crop
  top_cutoff = 40 # cut off world beyond the track
  new_height = 40
  frame = frame[top_cutoff:top_cutoff+new_height, :]

  # make it square
  square_dim = (120,120)
  frame = cv2.resize(frame, square_dim, interpolation = cv2.INTER_AREA)

  # apply car mask
  frame = cv2.bitwise_and(frame,frame,mask = car_mask)
  frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

  # BGR -> RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # rescale
  frame = preprocess_input(frame)
  frame = tf.expand_dims(frame, axis=0)

  return frame

async def main(context: Context):
    config_manager = ConfigurationManager()
    conf = config_manager.config
    # transformer = Transformer(conf)
    # recorder = Recorder(conf)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    control_mode = conf.control_mode
    dagger_training_enabled = conf.dagger_training_enabled
    dagger_epoch_size = conf.dagger_epoch_size

    new_model = tf.keras.models.load_model('better_model.h5')
    #new_model.summary()
    print("i love rcsnail")

    try:
        await initialize_subscriber(data_queue, conf.data_queue_port)
        await initialize_publisher(controls_queue, conf.controls_queue_port)

        frame_num = 0

        next_controls = None

        old_frame = np.zeros((1,224,224,3))

        while True:
            frame_num += 1
            
            frame, data = await recv_array_with_json(queue=data_queue)
            start_thinking = time.time()

            telemetry, expert_action = data
            if frame is None or telemetry is None or expert_action is None:
                logging.info("None data")
                continue

            try:
                # next_controls = expert_action.copy()

                if frame_num % 8 == 0:
                    print(frame_num)

                    frame = process_frame(frame)              
                    steering = new_model.predict(frame, steps = 1)
                    direction = (np.argmax(steering[0], axis = 0) -1)*0.75
                    # print(direction)
                    direction = float(direction)

                    next_controls = {"p":16565,"c":1593708369816,"g":1,"s":direction,"t":0.5,"b":0}

                    reverse = drive_backwards(frame, old_frame)
                    if (reverse == True):
                        time_start_reverse = time.time()
                        a = time.time() - time_start_reverse
                        while (a < 3):
                            print("loop reverse")
                            print(a)
                            next_controls = {"p":16565,"c":1593708369816,"g":-1,"s":0.0,"t":1.0,"b":0}
                            a = time.time() - time_start_reverse

                    old_frame = copy.deepcopy(frame)
                
                # recorder.record_full(frame, telemetry, expert_action, next_controls)
                controls_queue.send_json(next_controls)

            except Exception as ex:
                print("Sending exception: {}".format(ex))
                traceback.print_tb(ex.__traceback__)

    except Exception as ex:
        print("Exception: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        data_queue.close()
        controls_queue.close()

        # if recorder is not None:
        #     recorder.save_session_with_expert()


def cancel_tasks(loop):
    for task in asyncio.Task.all_tasks(loop):
        task.cancel()

def signal_cancel_tasks(*args):
    loop = asyncio.get_event_loop()
    for task in asyncio.Task.all_tasks(loop):
        task.cancel()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    #real_path = os.path.realpath(__file__)
    #dir_path = os.path.dirname(real_path)
    #model1 = keras.models.load_model(dir_path + "\model_v1")

    #model2 = keras.models.load_model('.\model_2.h5')#('path/to/location')
    #model2.summary()

    loop = asyncio.get_event_loop()
    # not implemented in Windows:
    # loop.add_signal_handler(signal.SIGINT, cancel_tasks, loop)
    # loop.add_signal_handler(signal.SIGTERM, cancel_tasks, loop)
    # alternative
    signal.signal(signal.SIGINT, signal_cancel_tasks)
    signal.signal(signal.SIGTERM, signal_cancel_tasks)

    context = zmq.asyncio.Context()
    try:
        loop.run_until_complete(main(context))
    except Exception as ex:
        logging.error("Base interruption: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        loop.close()
        context.destroy()