import asyncio
from bleak import BleakScanner

async def scan_all(timeout: float = 10.0):
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        print(f"{d.address}  name={d.name!r}")

if __name__ == "__main__":
    asyncio.run(scan_all())
