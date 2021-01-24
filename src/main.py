import time
import logging
import traceback
import asyncio
import signal
import numpy as np
import zmq
import cv2
from zmq.asyncio import Context

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager
# from utilities.recorder import Recorder

SKIP_FRAMES = 8 # we make decision every 9th frame, should be a multiple of (SKIP_FRAMES_BUFFER+1)
SKIP_FRAMES_BUFFER = 2 # we put every 3rd frame into a buffer
BUFFER_SIZE = 3
PATH_TO_CAR_MASK = 'new-mask-car-1.png'
PATH_TO_MODEL = 'fine-tuned-classifier.h5'
CONSTANT_THROTTLE = 0.55
CONSTANT_THROTTLE_REVERSE = 1.0
TIME_BACKWARDS = 0.6

def should_drive_backwards(frame, old_frame):
    dif = np.sum(np.absolute(np.array(frame) - np.array(old_frame)))
    if (dif > 1000):
        return False
    else:
        return True

class MaxSizeList(object): # from here: https://codereview.stackexchange.com/a/159065
    def __init__(self, size_limit):
        self.list = [None] * size_limit
        self.next = 0

    def push(self, item):
        self.list[self.next % len(self.list)] = item
        self.next += 1

    def get_list(self):
        if self.next < len(self.list):
            return self.list[:self.next]
        else:
            split = self.next % len(self.list)
            return self.list[split:] + self.list[:split]

    def __len__(self):
        return len(self.get_list())

def process_frame_car2(frame):
  
  # crop
  top_cutoff = frame.shape[0]//2 # cut off world beyond the track
  new_height = top_cutoff
  frame = frame[(top_cutoff - 6):(top_cutoff+new_height - 6), :]
  
  # apply car mask
  frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
  frame = cv2.bitwise_and(frame,frame,mask = car_mask)

  # BGR -> RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  #cv2.imshow('image', frame)
  #cv2.waitKey(0)

  # rescale
  frame = preprocess_input(frame)
  frame = tf.expand_dims(frame, axis=0)

  return frame

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

  #cv2.imshow('image', frame)
  #cv2.waitKey(0)

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

        frame_num = -1

        next_controls = None

        buffer = MaxSizeList(BUFFER_SIZE)     
            
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
                
                should_use_frame = frame_num % (SKIP_FRAMES_BUFFER+1) == SKIP_FRAMES_BUFFER
                if should_use_frame:
                    frame = process_frame(frame)
                    buffer.push(frame)

                if len(buffer) >= BUFFER_SIZE and frame_num % SKIP_FRAMES+1 == SKIP_FRAMES:
                    list_images = buffer.get_list()
                    previous_image, last_image = list_images[-2:]

                    steering = decision_model.predict(last_image, steps = 1)
                    direction = (np.argmax(steering[0], axis = 0) -1)*0.75
                    direction = float(direction)
                    direction -= 0.12

                    next_controls = {"p":packet_num,"c":timestamp,"g":1,"s":direction,"t":CONSTANT_THROTTLE,"b":0}

                    if should_drive_backwards(last_image, previous_image):
                        next_controls = {"p":packet_num,"c":timestamp,"g":-1,"s":0.0,"t":CONSTANT_THROTTLE_REVERSE,"b":0}
                        controls_queue.send_json(next_controls) 
                
                if abs(expert_action['t']) > 0:
                    next_controls = expert_action.copy() # manual controls
                    next_controls['t'] = CONSTANT_THROTTLE + 0.001
                    print('Expert intervention', next_controls) 

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

    context = Context()#zmq.asyncio.Context()
    try:
        loop.run_until_complete(main(context))
    except Exception as ex:
        logging.error("Base interruption: {}".format(ex))
        traceback.print_tb(ex.__traceback__)
    finally:
        loop.close()
        context.destroy()