import asyncio
import http
import logging
import time
import traceback

from web_infer_utils.openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from web_infer_utils.MVActor import MVActor
import numpy as np


logger = logging.getLogger(__name__)


class MVActorServer(MVActor):
    
    def __init__(self, host, port, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if obs["prompt"].find("<reset>")>=0:
                    self.reset()
                obs["prompt"] = obs["prompt"].replace("<reset>", "")

                infer_time = time.monotonic()

                ###
                ### obs:
                ###     state          : state
                ###     prompt         : task instruction. If prompt startswith '<reset>', the server will reset all saved memories first.
                ###     obs            : images, 
                ###     execution_step : number of execution steps

                action = self.play(**obs)

                action = dict(actions=action,)

                infer_time = time.monotonic() - infer_time

                await websocket.send(packer.pack(action))
                
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break

            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
