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
# from utilities.recorder import Recorder

SKIP_FRAMES = 8
PATH_TO_CAR_MASK = 'car-mask-224x224.png'
PATH_TO_MODEL = 'classifier-wheel-dataset.h5'
CONSTANT_THROTTLE = 0.55

def process_frame(frame):
  
  # crop
  top_cutoff = frame.shape[0]//2 # cut off world beyond the track
  new_height = top_cutoff
  frame = frame[top_cutoff:top_cutoff+new_height, :]
  
  # apply car mask
  frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
  frame = cv2.bitwise_and(frame,frame,mask = car_mask)

  # BGR -> RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # rescale
  frame = preprocess_input(frame)
  frame = tf.expand_dims(frame, axis=0)

  return frame

async def main(context: Context):
    global car_mask

    config_manager = ConfigurationManager()
    conf = config_manager.config
    # recorder = Recorder(conf)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    decision_model = tf.keras.models.load_model(PATH_TO_MODEL)
    car_mask = cv2.imread(PATH_TO_CAR_MASK,0)

    print("Car READY")

    try:
        await initialize_subscriber(data_queue, conf.data_queue_port)
        await initialize_publisher(controls_queue, conf.controls_queue_port)

        frame_num = 0

        next_controls = None

        while True:
            frame_num += 1
            
            frame, data = await recv_array_with_json(queue=data_queue)

            telemetry, expert_action = data
            packet_num = expert_action['p']
            timestamp = expert_action['c']
            if frame is None or telemetry is None or expert_action is None:
                logging.info("None data")
                continue

            try:
                # next_controls = expert_action.copy() # manual controls

                if frame_num % SKIP_FRAMES == 1:
                    frame = process_frame(frame)              
                    steering = decision_model.predict(frame, steps = 1)
                    direction = (np.argmax(steering[0], axis = 0) -1)*0.75
                    direction = float(direction)
                    direction -= 0.12

                    next_controls = {"p":packet_num,"c":timestamp,"g":1,"s":direction,"t":CONSTANT_THROTTLE,"b":0}

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