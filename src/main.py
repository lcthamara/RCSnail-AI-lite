import time
import logging
import traceback
import asyncio
import signal
import numpy as np
import zmq
from zmq.asyncio import Context

from commons.common_zmq import recv_array_with_json, initialize_subscriber, initialize_publisher
from commons.configuration_manager import ConfigurationManager

# from src.utilities.transformer import Transformer
from utilities.recorder import Recorder


async def main(context: Context):
    config_manager = ConfigurationManager()
    conf = config_manager.config
    # transformer = Transformer(conf)
    recorder = Recorder(conf)

    data_queue = context.socket(zmq.SUB)
    controls_queue = context.socket(zmq.PUB)

    control_mode = conf.control_mode
    dagger_training_enabled = conf.dagger_training_enabled
    dagger_epoch_size = conf.dagger_epoch_size

    try:
        mem_slice_frames = []
        mem_slice_numerics = []
        data_count = 0
        
        await initialize_subscriber(data_queue, conf.data_queue_port)
        await initialize_publisher(controls_queue, conf.controls_queue_port)

        while True:
            frame, data = await recv_array_with_json(queue=data_queue)
            telemetry, expert_action = data
            if frame is None or telemetry is None or expert_action is None:
                logging.info("None data")
                continue

            try:
                next_controls = expert_action.copy()
                time.sleep(0.01)
                
                recorder.record_full(frame, telemetry, expert_action, next_controls)
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

        if recorder is not None:
            recorder.save_session_with_expert()


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
