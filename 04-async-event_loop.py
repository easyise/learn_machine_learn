import asyncio

async def worker(name):
    print(f"Старт {name}")

    await asyncio.sleep(2)

    print(f"Конец {name}")


asyncio.run(worker("Worker 1"))


