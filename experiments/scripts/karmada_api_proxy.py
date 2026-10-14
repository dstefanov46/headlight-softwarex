#!/usr/bin/env python3
"""Small TCP passthrough used to expose Karmada on stable localhost TLS."""

from __future__ import annotations

import argparse
import asyncio
import contextlib


async def copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return
    await asyncio.gather(
        copy_stream(client_reader, target_writer),
        copy_stream(target_reader, client_writer),
    )


async def serve(listen_port: int, target_host: str, target_port: int) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, target_host, target_port),
        host="0.0.0.0",
        port=listen_port,
    )
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, default=32443)
    parser.add_argument("--target-host", default="karmada-host-control-plane")
    parser.add_argument("--target-port", type=int, default=32443)
    args = parser.parse_args()
    asyncio.run(serve(args.listen_port, args.target_host, args.target_port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
